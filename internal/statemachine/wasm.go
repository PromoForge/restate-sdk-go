package statemachine

import (
	"context"
	_ "embed"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/restatedev/sdk-go/internal/errors"
	pbinternal "github.com/restatedev/sdk-go/internal/generated"
	loginternal "github.com/restatedev/sdk-go/internal/log"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"google.golang.org/protobuf/proto"
)

// -- WASM core initialization

//go:embed shared_core_golang_wasm_binding.wasm
var sharedCore []byte

var _sharedCoreMod wazero.CompiledModule
var _wazeroRuntime wazero.Runtime

// Idle cores wait in two tiers. idleCores survives GC, so a service whose invocations are minutes
// apart does not instantiate a fresh module for each one; modPool takes the overflow of a burst,
// and GC drops what it holds.
var idleCores = make([]idleSlot, runtime.GOMAXPROCS(0))
var modPool sync.Pool

// idleSlot is padded to a cache line so that invocations ending on different CPUs do not contend.
type idleSlot struct {
	core atomic.Pointer[Core]
	_    [56]byte
}

func takeIdleCore() *Core {
	n := len(idleCores)
	start := int(rand.Uint32N(uint32(n)))
	for i := range n {
		slot := &idleCores[(start+i)%n]
		if core := slot.core.Load(); core != nil && slot.core.CompareAndSwap(core, nil) {
			return core
		}
	}
	if core, ok := modPool.Get().(*Core); ok {
		return core
	}
	return nil
}

func putIdleCore(core *Core) {
	n := len(idleCores)
	start := int(rand.Uint32N(uint32(n)))
	for i := range n {
		slot := &idleCores[(start+i)%n]
		if slot.core.Load() == nil && slot.core.CompareAndSwap(nil, core) {
			return
		}
	}
	modPool.Put(core)
}

// maxPooledCoreMemory bounds the linear memory a core may have grown to and still be kept idle.
// Wasm memory never shrinks, so a core that served a large payload is closed instead.
const maxPooledCoreMemory = 4 << 20

var coreTraceLogging = strings.ToLower(os.Getenv("CORE_TRACE_LOGGING_ENABLED"))

// compileSharedCore builds the wazero runtime and compiles the shared core the first time a core is
// needed, rather than at package init. Compiling takes hundreds of milliseconds, and every program
// that links the SDK paid it at start-up, including ones that never run a handler.
var compileSharedCore = sync.OnceValue(func() error {
	ctx := context.Background()

	_wazeroRuntime = wazero.NewRuntime(ctx)
	_, err := _wazeroRuntime.NewHostModuleBuilder("env").
		NewFunctionBuilder().WithFunc(wasmLogExport).Export("log").
		Instantiate(ctx)
	if err != nil {
		return fmt.Errorf("cannot instantiate the host module: %w", err)
	}

	_sharedCoreMod, err = _wazeroRuntime.CompileModule(ctx, sharedCore)
	if err != nil {
		return fmt.Errorf("cannot compile shared core WASM module: %w", err)
	}
	return nil
})

// -- Exported logging

const (
	abiLogLevelTrace uint32 = 0
	abiLogLevelDebug uint32 = 1
	abiLogLevelInfo  uint32 = 2
	abiLogLevelWarn  uint32 = 3
	abiLogLevelError uint32 = 4
)

func wasmLogExport(ctx context.Context, m api.Module, level, offset, byteCount uint32) {
	logger := getLogger(ctx)
	if logger == nil {
		logger = slog.Default()
	}
	switch level {
	case abiLogLevelTrace:
		// Access memory only if the log level is enabled
		buf, ok := m.Memory().Read(offset, byteCount)
		if !ok {
			panic(fmt.Sprintf("Memory.Read(%d, %d) out of range", offset, byteCount))
		}
		logger.Log(ctx, loginternal.LevelTrace, string(buf))
		break
	case abiLogLevelDebug:
		// Access memory only if the log level is enabled
		buf, ok := m.Memory().Read(offset, byteCount)
		if !ok {
			panic(fmt.Sprintf("Memory.Read(%d, %d) out of range", offset, byteCount))
		}
		logger.Debug(string(buf))
		break
	case abiLogLevelInfo:
		// Access memory only if the log level is enabled
		buf, ok := m.Memory().Read(offset, byteCount)
		if !ok {
			panic(fmt.Sprintf("Memory.Read(%d, %d) out of range", offset, byteCount))
		}
		logger.Info(string(buf))
		break
	case abiLogLevelWarn:
		// Access memory only if the log level is enabled
		buf, ok := m.Memory().Read(offset, byteCount)
		if !ok {
			panic(fmt.Sprintf("Memory.Read(%d, %d) out of range", offset, byteCount))
		}
		logger.Warn(string(buf))
		break
	case abiLogLevelError:
		// Access memory only if the log level is enabled
		buf, ok := m.Memory().Read(offset, byteCount)
		if !ok {
			panic(fmt.Sprintf("Memory.Read(%d, %d) out of range", offset, byteCount))
		}
		logger.Error(string(buf))
		break
	}
}

// -- Exposed API

type Core struct {
	mod api.Module

	// This is used to make sure the core is not concurrently accessed
	coreMutex sync.Mutex

	// inUse is true from NewCore until Close.
	inUse atomic.Bool
	// owner is the state machine whose VM lives in this core, nil once it is freed.
	owner atomic.Pointer[StateMachine]
	// poisoned is set when a call into the module trapped or panicked: its memory may be
	// inconsistent, so the core is closed instead of reused.
	poisoned *atomic.Bool

	// Used to avoid allocating on each Call
	callStack []uint64

	allocate   api.Function
	deallocate api.Function

	vmNew                  api.Function
	vmGetResponseHead      api.Function
	vmNotifyInput          api.Function
	vmNotifyInputClosed    api.Function
	vmNotifyError          api.Function
	vmIsReadyToExecute     api.Function
	vmIsCompleted          api.Function
	vmIsProcessing         api.Function
	vmDoProgress           api.Function
	vmTakeNotification     api.Function
	vmTakeOutput           api.Function
	vmSysInput             api.Function
	vmSysStateGet          api.Function
	vmSysStateGetKeys      api.Function
	vmSysStateSet          api.Function
	vmSysStateClear        api.Function
	vmSysStateClearAll     api.Function
	vmSysSleep             api.Function
	vmSysAwakeable         api.Function
	vmSysCompleteAwakeable api.Function
	vmSysCall              api.Function
	vmSysSend              api.Function
	vmSysCancelInvocation  api.Function
	vmSysAttachInvocation  api.Function
	vmSysPromiseGet        api.Function
	vmSysPromisePeek       api.Function
	vmSysPromiseComplete   api.Function
	vmSysRun               api.Function
	vmProposeRunCompletion api.Function
	vmSysWriteOutput       api.Function
	vmSysEnd               api.Function
	vmFree                 api.Function
}

func NewCore(ctx context.Context) (*Core, error) {
	if core := takeIdleCore(); core != nil {
		core.inUse.Store(true)
		return core, nil
	}

	if err := compileSharedCore(); err != nil {
		return nil, err
	}

	instance, err := _wazeroRuntime.InstantiateModule(
		ctx,
		_sharedCoreMod,
		wazero.NewModuleConfig().
			WithStartFunctions().
			WithName(""))
	if err != nil {
		return nil, fmt.Errorf("cannot instantiate new core: %e", err)
	}

	// Init module, this sets up few things such as the panic handler
	logPreFiltering := abiLogLevelDebug
	if coreTraceLogging == "true" {
		logPreFiltering = abiLogLevelTrace
	}
	_, err = instance.ExportedFunction("init").Call(ctx, uint64(logPreFiltering))
	if err != nil {
		instance.Close(ctx)
		return nil, fmt.Errorf("cannot instantiate new core: %e", err)
	}

	poisoned := new(atomic.Bool)
	export := func(name string) api.Function {
		return guardedFunction{Function: instance.ExportedFunction(name), poisoned: poisoned}
	}
	core := &Core{
		mod:                    instance,
		poisoned:               poisoned,
		callStack:              make([]uint64, 4),
		allocate:               export("allocate"),
		deallocate:             export("deallocate"),
		vmNew:                  export("vm_new"),
		vmGetResponseHead:      export("vm_get_response_head"),
		vmNotifyInput:          export("vm_notify_input"),
		vmNotifyInputClosed:    export("vm_notify_input_closed"),
		vmNotifyError:          export("vm_notify_error"),
		vmIsReadyToExecute:     export("vm_is_ready_to_execute"),
		vmIsCompleted:          export("vm_is_completed"),
		vmIsProcessing:         export("vm_is_processing"),
		vmDoProgress:           export("vm_do_progress"),
		vmTakeNotification:     export("vm_take_notification"),
		vmTakeOutput:           export("vm_take_output"),
		vmSysInput:             export("vm_sys_input"),
		vmSysStateGet:          export("vm_sys_state_get"),
		vmSysStateGetKeys:      export("vm_sys_state_get_keys"),
		vmSysStateSet:          export("vm_sys_state_set"),
		vmSysStateClear:        export("vm_sys_state_clear"),
		vmSysStateClearAll:     export("vm_sys_state_clear_all"),
		vmSysSleep:             export("vm_sys_sleep"),
		vmSysAwakeable:         export("vm_sys_awakeable"),
		vmSysCompleteAwakeable: export("vm_sys_complete_awakeable"),
		vmSysCall:              export("vm_sys_call"),
		vmSysSend:              export("vm_sys_send"),
		vmSysCancelInvocation:  export("vm_sys_cancel_invocation"),
		vmSysAttachInvocation:  export("vm_sys_attach_invocation"),
		vmSysPromiseGet:        export("vm_sys_promise_get"),
		vmSysPromisePeek:       export("vm_sys_promise_peek"),
		vmSysPromiseComplete:   export("vm_sys_promise_complete"),
		vmSysRun:               export("vm_sys_run"),
		vmProposeRunCompletion: export("vm_propose_run_completion"),
		vmSysWriteOutput:       export("vm_sys_write_output"),
		vmSysEnd:               export("vm_sys_end"),
		vmFree:                 export("vm_free"),
	}

	core.inUse.Store(true)

	// Closes the WASM module of a core dropped by GC: from modPool, or never closed.
	runtime.AddCleanup(core, func(instance api.Module) {
		instance.Close(context.Background())
	}, instance)

	return core, nil
}

// Close ends the core's use by an invocation. The core is kept for the next invocation only if its
// state machine was freed and no call into the module failed: the shared core then holds no
// per-invocation state, as its VM, the only such state, was dropped by vm_free. Otherwise, or for
// a core that grew past maxPooledCoreMemory, the module is closed.
func (core *Core) Close(ctx context.Context) error {
	if !core.inUse.CompareAndSwap(true, false) {
		return nil
	}
	reusable := core.owner.Load() == nil &&
		!core.poisoned.Load() &&
		core.mod.Memory().Size() <= maxPooledCoreMemory
	if !reusable {
		return core.mod.Close(ctx)
	}
	putIdleCore(core)
	return nil
}

// guardedFunction poisons its core when a call into the module fails, which for the shared core
// means it trapped.
type guardedFunction struct {
	api.Function
	poisoned *atomic.Bool
}

func (f guardedFunction) Call(ctx context.Context, params ...uint64) ([]uint64, error) {
	res, err := f.Function.Call(ctx, params...)
	if err != nil {
		f.poisoned.Store(true)
	}
	return res, err
}

func (f guardedFunction) CallWithStack(ctx context.Context, stack []uint64) error {
	err := f.Function.CallWithStack(ctx, stack)
	if err != nil {
		f.poisoned.Store(true)
	}
	return err
}

type concurrentContextUseError struct{}

func (c concurrentContextUseError) Error() string {
	return "Concurrent context use detected; either a Context method was used while a Run() is in progress, or Context methods are being called from multiple goroutines. Failing invocation."
}

type stateMachineFreedError struct{}

func (stateMachineFreedError) Error() string {
	return "Context used after its invocation ended; the state machine was already freed."
}

func (core *Core) lock() {
	if !core.coreMutex.TryLock() {
		panic(concurrentContextUseError{})
	}
}

// unlock must be deferred directly, so that recover sees a panic raised while the core was held.
// Such a panic may leave the module's memory half-written, so it poisons the core.
func (core *Core) unlock() {
	if r := recover(); r != nil {
		core.poisoned.Store(true)
		core.coreMutex.Unlock()
		panic(r)
	}
	core.coreMutex.Unlock()
}

// enter takes the core for one call, refusing a state machine that was freed: its core may already
// serve another invocation.
func (sm *StateMachine) enter() {
	if sm.core.owner.Load() != sm {
		panic(stateMachineFreedError{})
	}
	sm.core.lock()
	if sm.core.owner.Load() != sm {
		sm.core.coreMutex.Unlock()
		panic(stateMachineFreedError{})
	}
}

func (core *Core) NewStateMachine(ctx context.Context, headers []*pbinternal.Header) (*StateMachine, error) {
	core.lock()
	defer core.unlock()
	if core.owner.Load() != nil {
		return nil, fmt.Errorf("the core already holds a state machine")
	}

	params := pbinternal.VmNewParameters{}
	params.SetHeaders(headers)
	inputPtr, inputLen := core.transferInputStructToWasmMemory(ctx, &params)

	core.callStack[0] = inputPtr
	core.callStack[1] = inputLen
	err := core.vmNew.CallWithStack(ctx, core.callStack)
	out := core.callStack[0]
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_new: %e", err)
	}

	output := pbinternal.VmNewReturn{}
	core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return nil, wasmFailureToGoError(output.GetFailure())
	}
	sm := &StateMachine{
		core:      core,
		vmPointer: uint64(output.GetPointer())}
	core.owner.Store(sm)
	return sm, nil
}

type StateMachine struct {
	core      *Core
	vmPointer uint64
}

func (sm *StateMachine) GetResponseHead(ctx context.Context) (*pbinternal.VmGetResponseHeadReturn, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmGetResponseHead.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_get_response_head: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmGetResponseHeadReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	return &output, nil
}

func (sm *StateMachine) NotifyInputClosed(ctx context.Context) error {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmNotifyInputClosed.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_notify_input_closed: %e", err)
	}
	return nil
}

func (sm *StateMachine) NotifyInput(ctx context.Context, bytes []byte) error {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferBytesToWasmMemory(ctx, bytes)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmNotifyInput.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_notify_input: %e", err)
	}

	return nil
}

func (sm *StateMachine) NotifyError(ctx context.Context, message string, stacktrace string) error {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmNotifyError{}
	params.SetMessage(message)
	params.SetStacktrace(stacktrace)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmNotifyError.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_notify_error: %e", err)
	}

	return nil
}

func (sm *StateMachine) IsReadyToExecute(ctx context.Context) (bool, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmIsReadyToExecute.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return false, fmt.Errorf("error when calling vm_is_ready_to_execute: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmIsReadyToExecuteReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return false, wasmFailureToGoError(output.GetFailure())
	}

	return output.GetReady(), nil
}

func (sm *StateMachine) IsCompleted(ctx context.Context, handle uint32) (bool, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = uint64(handle)
	err := sm.core.vmIsCompleted.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return false, fmt.Errorf("error when calling vm_is_completed: %e", err)
	}

	if sm.core.callStack[0] == 0 {
		return false, nil
	}
	return true, nil
}

func (sm *StateMachine) IsProcessing(ctx context.Context) (bool, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmIsProcessing.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return false, fmt.Errorf("error when calling vm_is_processing: %e", err)
	}

	if sm.core.callStack[0] == 0 {
		return false, nil
	}
	return true, nil
}

type SuspensionError struct{}

func (s SuspensionError) Error() string {
	return "Suspended"
}

type DoProgressResult interface {
	isDoProgressResult()
}

type DoProgressAnyCompleted struct{}

func (DoProgressAnyCompleted) isDoProgressResult() {}

type DoProgressReadFromInput struct{}

func (DoProgressReadFromInput) isDoProgressResult() {}

type DoProgressCancelSignalReceived struct{}

func (DoProgressCancelSignalReceived) isDoProgressResult() {}

type DoProgressWaitingPendingRun struct{}

func (DoProgressWaitingPendingRun) isDoProgressResult() {}

type DoProgressExecuteRun struct {
	Handle uint32
}

func (DoProgressExecuteRun) isDoProgressResult() {}

func (sm *StateMachine) DoProgress(ctx context.Context, handles []uint32) (DoProgressResult, error) {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmDoProgressParameters{}
	params.SetHandles(handles)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmDoProgress.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_do_progress: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmDoProgressReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return nil, wasmFailureToGoError(output.GetFailure())
	}
	if output.HasSuspended() {
		return nil, SuspensionError{}
	}
	if output.HasAnyCompleted() {
		return DoProgressAnyCompleted{}, nil
	}
	if output.HasReadFromInput() {
		return DoProgressReadFromInput{}, nil
	}
	if output.HasCancelSignalReceived() {
		return DoProgressCancelSignalReceived{}, nil
	}
	if output.HasWaitingPendingRun() {
		return DoProgressWaitingPendingRun{}, nil
	}
	if output.HasExecuteRun() {
		return DoProgressExecuteRun{
			Handle: output.GetExecuteRun(),
		}, nil
	}
	panic("Missing result")
}

type Value interface {
	isValue()
}

type ValueVoid struct{}

func (ValueVoid) isValue() {}

type ValueSuccess struct {
	Success []byte
}

func (ValueSuccess) isValue() {}

type ValueFailure struct {
	Failure *pbinternal.Failure
}

func (ValueFailure) isValue() {}

type ValueStateKeys struct {
	Keys []string
}

func (ValueStateKeys) isValue() {}

type ValueInvocationId struct {
	InvocationId string
}

func (ValueInvocationId) isValue() {}

type ValueExecuteRun struct {
	Handle uint32
}

func (ValueExecuteRun) isValue() {}

func (sm *StateMachine) TakeNotification(ctx context.Context, handle uint32) (Value, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = uint64(handle)
	err := sm.core.vmTakeNotification.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_take_notification: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmTakeNotificationReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return nil, wasmFailureToGoError(output.GetFailure())
	}
	if output.HasSuspended() {
		return nil, SuspensionError{}
	}
	if output.HasNotReady() {
		return nil, nil
	}

	value := output.GetValue()
	if value.HasVoid() {
		return ValueVoid{}, nil
	}
	if value.HasSuccess() {
		return ValueSuccess{Success: value.GetSuccess()}, nil
	}
	if value.HasFailure() {
		return ValueFailure{
			Failure: value.GetFailure(),
		}, nil
	}
	if value.HasStateKeys() {
		return ValueStateKeys{
			Keys: value.GetStateKeys().GetKeys(),
		}, nil
	}
	if value.HasInvocationId() {
		return ValueInvocationId{
			InvocationId: value.GetInvocationId(),
		}, nil
	}
	panic("Missing result")
}

func (sm *StateMachine) TakeOutput(ctx context.Context) ([]byte, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmTakeOutput.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_take_output: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmTakeOutputReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasEOF() {
		return nil, io.EOF
	}

	return output.GetBytes(), nil
}

func (sm *StateMachine) SysInput(ctx context.Context) (*pbinternal.VmSysInputReturn_Input, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmSysInput.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return nil, fmt.Errorf("error when calling vm_sys_input: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmSysInputReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return nil, wasmFailureToGoError(output.GetFailure())
	}

	return output.GetOk(), nil
}

func (sm *StateMachine) SysStateGet(ctx context.Context, input *pbinternal.VmSysStateGetParameters) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysStateGet.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_state_get: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysStateGetKeys(ctx context.Context) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmSysStateGetKeys.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_state_get_keys: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysStateSet(ctx context.Context, input *pbinternal.VmSysStateSetParameters) error {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysStateSet.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_state_set: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysStateClear(ctx context.Context, key string) error {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysStateClearParameters{}
	params.SetKey(key)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysStateClear.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_state_clear: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysStateClearAll(ctx context.Context) error {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmSysStateClearAll.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_state_clear_all: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmSysInputReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}

	return nil
}

func (sm *StateMachine) SysSleep(ctx context.Context, name string, duration time.Duration) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	now := time.Now()
	params := pbinternal.VmSysSleepParameters{}
	params.SetWakeUpTimeSinceUnixEpochMillis(uint64(now.Add(duration).UnixMilli()))
	params.SetNowSinceUnixEpochMillis(uint64(now.UnixMilli()))
	params.SetName(name)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysSleep.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_sleep: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysAwakeable(ctx context.Context) (string, uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmSysAwakeable.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return "", 0, fmt.Errorf("error when calling vm_sys_awakable: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmSysAwakeableReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return "", 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetOk().GetId(), output.GetOk().GetHandle(), nil
}

func (sm *StateMachine) SysCall(ctx context.Context, input *pbinternal.VmSysCallParameters) (uint32, uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysCall.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, 0, fmt.Errorf("error when calling vm_sys_call: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.VmSysCallReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetOk().GetInvocationIdHandle(), output.GetOk().GetResultHandle(), nil
}

func (sm *StateMachine) SysSend(ctx context.Context, input *pbinternal.VmSysSendParameters) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysSend.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_send: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysCancelInvocation(ctx context.Context, invocationId string) error {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysCancelInvocation{}
	params.SetInvocationId(invocationId)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysCancelInvocation.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_cancel_invocation: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysAttachInvocation(ctx context.Context, invocationId string) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysAttachInvocation{}
	params.SetInvocationId(invocationId)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysAttachInvocation.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_attach_invocation: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysCompleteAwakeable(ctx context.Context, input *pbinternal.VmSysCompleteAwakeableParameters) error {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysCompleteAwakeable.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_complete_awakeable: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysPromiseGet(ctx context.Context, key string) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysPromiseGetParameters{}
	params.SetKey(key)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysPromiseGet.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_promise_get: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysPromisePeek(ctx context.Context, key string) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysPromisePeekParameters{}
	params.SetKey(key)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysPromisePeek.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_promise_peek: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysPromiseComplete(ctx context.Context, input *pbinternal.VmSysPromiseCompleteParameters) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysPromiseComplete.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_promise_complete: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) SysRun(ctx context.Context, name string) (uint32, error) {
	sm.enter()
	defer sm.core.unlock()

	params := pbinternal.VmSysRunParameters{}
	params.SetName(name)
	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, &params)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysRun.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return 0, fmt.Errorf("error when calling vm_sys_run: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.SimpleSysAsyncResultReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return 0, wasmFailureToGoError(output.GetFailure())
	}
	return output.GetHandle(), nil
}

func (sm *StateMachine) ProposeRunCompletion(ctx context.Context, input *pbinternal.VmProposeRunCompletionParameters) error {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmProposeRunCompletion.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_propose_run_completion: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysWriteOutput(ctx context.Context, input *pbinternal.VmSysWriteOutputParameters) error {
	sm.enter()
	defer sm.core.unlock()

	inputPtr, inputLen := sm.core.transferInputStructToWasmMemory(ctx, input)

	sm.core.callStack[0] = sm.vmPointer
	sm.core.callStack[1] = inputPtr
	sm.core.callStack[2] = inputLen
	err := sm.core.vmSysWriteOutput.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_write_output: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) SysEnd(ctx context.Context) error {
	sm.enter()
	defer sm.core.unlock()

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmSysEnd.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_sys_end: %e", err)
	}
	out := sm.core.callStack[0]

	output := pbinternal.GenericEmptyReturn{}
	sm.core.transferOutputStructFromWasmMemory(ctx, out, &output)

	if output.HasFailure() {
		return wasmFailureToGoError(output.GetFailure())
	}
	return nil
}

func (sm *StateMachine) Free(ctx context.Context) error {
	sm.enter()
	defer sm.core.unlock()
	defer sm.core.owner.Store(nil)

	sm.core.callStack[0] = sm.vmPointer
	err := sm.core.vmFree.CallWithStack(ctx, sm.core.callStack)
	if err != nil {
		return fmt.Errorf("error when calling vm_free: %e", err)
	}
	return nil
}

// -- Memory tingling

func wasmFailureToGoError(failure *pbinternal.Failure) error {
	return &errors.CodeError{
		Code:  errors.Code(failure.GetCode()),
		Inner: fmt.Errorf("[%d] %s", failure.GetCode(), failure.GetMessage()),
	}
}

func (core *Core) transferInputStructToWasmMemory(ctx context.Context, t proto.Message) (uint64, uint64) {
	buffer, err := proto.Marshal(t)
	if err != nil {
		log.Panicln(err)
	}
	return core.transferBytesToWasmMemory(ctx, buffer)
}

func (core *Core) transferBytesToWasmMemory(ctx context.Context, b []byte) (uint64, uint64) {
	bufferLen := uint64(len(b))

	// Allocate the memory we need. The de-allocation happens in the shared core.
	core.callStack[0] = bufferLen
	err := core.allocate.CallWithStack(ctx, core.callStack)
	if err != nil {
		log.Panicln(err)
	}
	namePtr := core.callStack[0]

	// The pointer is a linear memory offset, which is where we write the name.
	if !core.mod.Memory().Write(uint32(namePtr), b) {
		log.Panicf("Memory.Write(%d, %d) out of range of memory size %d", namePtr, bufferLen, core.mod.Memory().Size())
	}

	return namePtr, bufferLen
}

func (core *Core) transferOutputStructFromWasmMemory(ctx context.Context, packedPtr uint64, v proto.Message) {
	buffer := core.transferBytesFromWasmMemory(ctx, packedPtr)
	err := proto.Unmarshal(buffer, v)
	if err != nil {
		log.Panicln(fmt.Errorf("WTF buf: %s err: %e", string(buffer), err))
	}
}

func (core *Core) transferBytesFromWasmMemory(ctx context.Context, packedPtr uint64) []byte {
	resPtr := uint32(packedPtr >> 32)
	resSize := uint32(packedPtr)

	bytes, ok := core.mod.Memory().Read(resPtr, resSize)
	if !ok {
		log.Panicf("Memory.Read(%d, %d) out of range of memory size %d",
			resPtr, resSize, core.mod.Memory().Size())
	}
	tmp := make([]byte, len(bytes))
	copy(tmp, bytes)

	core.callStack[0] = uint64(resPtr)
	core.callStack[1] = uint64(resSize)
	err := core.deallocate.CallWithStack(ctx, core.callStack)
	if err != nil {
		log.Panicln(err)
	}

	return tmp
}
