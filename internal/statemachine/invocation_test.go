package statemachine

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"runtime"
	"testing"

	pbinternal "github.com/restatedev/sdk-go/internal/generated"
	"google.golang.org/protobuf/encoding/protowire"
)

const (
	startMessageType        uint16 = 0x0000
	inputCommandMessageType uint16 = 0x0400
)

func frame(ty uint16, body []byte) []byte {
	out := binary.BigEndian.AppendUint64(nil, uint64(ty)<<48|uint64(len(body)))
	return append(out, body...)
}

// invocationStream encodes the start message and the input command Restate sends to begin an
// invocation with the given id, key and input.
func invocationStream(id, key string, input []byte) []byte {
	var start []byte
	start = protowire.AppendTag(start, 1, protowire.BytesType)
	start = protowire.AppendBytes(start, []byte(id))
	start = protowire.AppendTag(start, 2, protowire.BytesType)
	start = protowire.AppendString(start, id)
	start = protowire.AppendTag(start, 3, protowire.VarintType)
	start = protowire.AppendVarint(start, 1)
	start = protowire.AppendTag(start, 6, protowire.BytesType)
	start = protowire.AppendString(start, key)

	var value []byte
	value = protowire.AppendTag(value, 1, protowire.BytesType)
	value = protowire.AppendBytes(value, input)
	var cmd []byte
	cmd = protowire.AppendTag(cmd, 14, protowire.BytesType)
	cmd = protowire.AppendBytes(cmd, value)

	return append(frame(startMessageType, start), frame(inputCommandMessageType, cmd)...)
}

func invocationHeaders() []*pbinternal.Header {
	h := &pbinternal.Header{}
	h.SetKey("content-type")
	h.SetValue("application/vnd.restate.invocation.v5")
	return []*pbinternal.Header{h}
}

type echoResult struct {
	key    string
	input  []byte
	output []byte
}

// runEcho drives one invocation through the core the way server.handleInvokeRequest does: a new
// state machine, the input stream, an echo handler, then Free and Close.
func runEcho(ctx context.Context, id, key string, input []byte) (echoResult, error) {
	core, err := NewCore(ctx)
	if err != nil {
		return echoResult{}, err
	}
	defer core.Close(ctx)

	sm, err := core.NewStateMachine(ctx, invocationHeaders())
	if err != nil {
		return echoResult{}, fmt.Errorf("NewStateMachine: %w", err)
	}
	defer sm.Free(ctx)

	if err := sm.NotifyInput(ctx, invocationStream(id, key, input)); err != nil {
		return echoResult{}, fmt.Errorf("NotifyInput: %w", err)
	}
	if err := sm.NotifyInputClosed(ctx); err != nil {
		return echoResult{}, fmt.Errorf("NotifyInputClosed: %w", err)
	}
	ready, err := sm.IsReadyToExecute(ctx)
	if err != nil {
		return echoResult{}, fmt.Errorf("IsReadyToExecute: %w", err)
	}
	if !ready {
		return echoResult{}, fmt.Errorf("state machine is not ready to execute")
	}

	in, err := sm.SysInput(ctx)
	if err != nil {
		return echoResult{}, fmt.Errorf("SysInput: %w", err)
	}
	res := echoResult{key: in.GetKey(), input: bytes.Clone(in.GetInput())}

	out := &pbinternal.VmSysWriteOutputParameters{}
	out.SetSuccess(in.GetInput())
	if err := sm.SysWriteOutput(ctx, out); err != nil {
		return echoResult{}, fmt.Errorf("SysWriteOutput: %w", err)
	}
	if err := sm.SysEnd(ctx); err != nil {
		return echoResult{}, fmt.Errorf("SysEnd: %w", err)
	}
	for {
		b, err := sm.TakeOutput(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return echoResult{}, fmt.Errorf("TakeOutput: %w", err)
		}
		res.output = append(res.output, b...)
	}
	return res, nil
}

func warmUp(b *testing.B) {
	if _, err := runEcho(context.Background(), "warm-up", "key", nil); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkNewCore(b *testing.B) {
	ctx := context.Background()
	warmUp(b)
	b.ReportAllocs()
	for b.Loop() {
		core, err := NewCore(ctx)
		if err != nil {
			b.Fatal(err)
		}
		core.Close(ctx)
	}
}

// BenchmarkNewCoreAfterGC takes a core after two GC cycles, as an idle service does at each tick.
func BenchmarkNewCoreAfterGC(b *testing.B) {
	ctx := context.Background()
	warmUp(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		runtime.GC()
		runtime.GC()
		b.StartTimer()
		core, err := NewCore(ctx)
		if err != nil {
			b.Fatal(err)
		}
		core.Close(ctx)
	}
}

func BenchmarkInvocation(b *testing.B) {
	ctx := context.Background()
	input := []byte("ping")
	warmUp(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := runEcho(ctx, "inv", "key", input); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInvocationParallel(b *testing.B) {
	ctx := context.Background()
	input := []byte("ping")
	warmUp(b)
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := runEcho(ctx, "inv", "key", input); err != nil {
				b.Error(err)
				return
			}
		}
	})
}

// BenchmarkInvocationAfterGC runs each invocation after two GC cycles, the shape of an idle
// service whose cron ticks are minutes apart: any GC-cleared cache is empty at every tick.
func BenchmarkInvocationAfterGC(b *testing.B) {
	ctx := context.Background()
	input := []byte("ping")
	warmUp(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		runtime.GC()
		runtime.GC()
		b.StartTimer()
		if _, err := runEcho(ctx, "inv", "key", input); err != nil {
			b.Fatal(err)
		}
	}
}
