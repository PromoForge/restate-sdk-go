package statemachine

import (
	"bytes"
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
)

func drainIdleCores(t *testing.T) {
	t.Helper()
	for core := takeIdleCore(); core != nil; core = takeIdleCore() {
		core.mod.Close(context.Background())
	}
}

func reservedCores() int {
	n := 0
	for i := range idleCores {
		if idleCores[i].core.Load() != nil {
			n++
		}
	}
	return n
}

func takeCore(t *testing.T) *Core {
	t.Helper()
	core, err := NewCore(context.Background())
	if err != nil {
		t.Fatalf("NewCore: %v", err)
	}
	return core
}

func TestCoreSurvivesGCBetweenInvocations(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	sm, err := core.NewStateMachine(ctx, invocationHeaders())
	if err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}
	if err := sm.Free(ctx); err != nil {
		t.Fatalf("Free: %v", err)
	}
	core.Close(ctx)

	runtime.GC()
	runtime.GC()

	if next := takeCore(t); next != core {
		t.Fatal("an idle core was dropped by GC; the next invocation instantiated a new module")
	}
}

func TestEchoOnAReusedCoreSeesOnlyItsOwnInvocation(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	first, err := runEcho(ctx, "inv-1", "key-1", []byte("first-input"))
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	second, err := runEcho(ctx, "inv-2", "key-2", []byte("2nd"))
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if second.key != "key-2" || string(second.input) != "2nd" {
		t.Fatalf("second invocation saw key %q input %q", second.key, second.input)
	}
	if bytes.Contains(second.output, []byte("first-input")) {
		t.Fatal("the second invocation's output carries the first invocation's input")
	}
	if !bytes.Contains(first.output, []byte("first-input")) || !bytes.Contains(second.output, []byte("2nd")) {
		t.Fatalf("outputs do not echo their inputs: %q, %q", first.output, second.output)
	}
}

func TestConcurrentInvocationsDoNotShareState(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()
	const workers, perWorker = 16, 50

	var wg sync.WaitGroup
	errs := make(chan error, workers*perWorker)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range perWorker {
				key := fmt.Sprintf("key-%d-%d", w, i)
				input := bytes.Repeat([]byte(fmt.Sprintf("w%di%d;", w, i)), 1+i%7)
				res, err := runEcho(ctx, fmt.Sprintf("inv-%d-%d", w, i), key, input)
				if err != nil {
					errs <- err
					continue
				}
				if res.key != key || !bytes.Equal(res.input, input) || !bytes.Contains(res.output, input) {
					errs <- fmt.Errorf("%s: got key %q input %q output %q", key, res.key, res.input, res.output)
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestATrappedCoreIsNotReused(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	sm, err := core.NewStateMachine(ctx, invocationHeaders())
	if err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}
	// A null input pointer makes the shared core panic, which traps the module.
	core.callStack[0], core.callStack[1], core.callStack[2] = sm.vmPointer, 0, 0
	if err := core.vmNotifyInput.CallWithStack(ctx, core.callStack); err == nil {
		t.Fatal("vm_notify_input with a null input did not trap")
	}
	_ = sm.Free(ctx)
	core.Close(ctx)

	if next := takeCore(t); next == core {
		t.Fatal("a core whose module trapped was handed to the next invocation")
	}
	if _, err := runEcho(ctx, "after-trap", "k", []byte("ok")); err != nil {
		t.Fatalf("invocation after a trapped one: %v", err)
	}
}

func TestAGoPanicWhileTheCoreIsHeldPoisonsIt(t *testing.T) {
	drainIdleCores(t)
	core := takeCore(t)

	func() {
		defer func() { _ = recover() }()
		core.lock()
		defer core.unlock()
		panic("half-written memory")
	}()
	if !core.poisoned.Load() {
		t.Fatal("a panic while the core was held did not poison it")
	}
	core.Close(context.Background())
	if next := takeCore(t); next == core {
		t.Fatal("a poisoned core was handed to the next invocation")
	}
}

func TestAnAbortedInvocationsCoreIsNotReused(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	if _, err := core.NewStateMachine(ctx, invocationHeaders()); err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}
	// Closed without freeing its state machine: its VM is still in the module's memory.
	core.Close(ctx)

	if next := takeCore(t); next == core {
		t.Fatal("a core still holding a VM was handed to the next invocation")
	}
	if _, err := runEcho(ctx, "after-abort", "k", []byte("ok")); err != nil {
		t.Fatalf("invocation after an aborted one: %v", err)
	}
}

func TestAFailedStateMachineStartLeavesTheCoreReusable(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	if _, err := core.NewStateMachine(ctx, nil); err == nil {
		t.Fatal("NewStateMachine without a content type succeeded")
	}
	core.Close(ctx)

	if next := takeCore(t); next != core {
		t.Fatal("a core whose vm_new failed cleanly was not reused")
	}
}

func TestAStateMachineUsedAfterFreeIsRefused(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	stale, err := core.NewStateMachine(ctx, invocationHeaders())
	if err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}
	if err := stale.Free(ctx); err != nil {
		t.Fatalf("Free: %v", err)
	}
	core.Close(ctx)

	next := takeCore(t)
	if next != core {
		t.Fatal("expected the freed core to be reused")
	}
	current, err := next.NewStateMachine(ctx, invocationHeaders())
	if err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}

	func() {
		defer func() {
			if _, ok := recover().(stateMachineFreedError); !ok {
				t.Fatal("a freed state machine reached the core now serving another invocation")
			}
		}()
		_ = stale.NotifyInputClosed(ctx)
	}()

	if err := current.NotifyInput(ctx, invocationStream("inv", "k", []byte("x"))); err != nil {
		t.Fatalf("the current invocation was disturbed: %v", err)
	}
	if err := current.Free(ctx); err != nil {
		t.Fatalf("Free: %v", err)
	}
	next.Close(ctx)
}

func TestACoreThatGrewPastTheBoundIsNotKept(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	if _, err := runEcho(ctx, "big", "k", make([]byte, 2*maxPooledCoreMemory)); err != nil {
		t.Fatalf("large invocation: %v", err)
	}
	if core := takeIdleCore(); core != nil {
		t.Fatalf("a core holding %d bytes of memory was kept idle", core.mod.Memory().Size())
	}
}

func TestCoresBeyondTheReserveOverflowToTheGCPool(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	cores := make([]*Core, len(idleCores)+2)
	for i := range cores {
		cores[i] = takeCore(t)
	}
	for _, core := range cores {
		core.Close(ctx)
	}
	if got := reservedCores(); got != len(idleCores) {
		t.Fatalf("%d cores in the reserve, want %d", got, len(idleCores))
	}

	seen := map[*Core]bool{}
	for range cores {
		core := takeCore(t)
		if seen[core] {
			t.Fatal("one idle core was handed out twice")
		}
		seen[core] = true
	}
}

func TestClosingACoreTwiceDoesNotPoolItTwice(t *testing.T) {
	drainIdleCores(t)
	ctx := context.Background()

	core := takeCore(t)
	core.Close(ctx)
	core.Close(ctx)
	if got := reservedCores(); got != 1 {
		t.Fatalf("%d idle cores after closing one core twice", got)
	}
	if takeCore(t) != core {
		t.Fatal("expected the closed core back")
	}
	if takeIdleCore() != nil {
		t.Fatal("a core closed twice was pooled twice")
	}
}
