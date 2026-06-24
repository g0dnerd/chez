// Helper worker for shared-memory Lazy SMP (chez-mt.wasm).
//
// Each helper is one extra instance of the threaded engine module, sharing the
// main thread's WebAssembly.Memory. Module-level state (the TT, the search
// state, the per-thread contexts) lives in that shared linear memory, so a
// helper only needs to (1) point its own __stack_pointer at the private stack
// region the main thread carved out for it, and (2) run its thread of the
// search. wasm globals are per-instance, so setting __stack_pointer here does
// not disturb the other instances.

let exports = null;

self.onmessage = async (e) => {
  const msg = e.data;
  switch (msg.cmd) {
    case "init": {
      // chez_now_ms must return absolute (epoch-based) ms so this worker's clock
      // agrees with the main thread's, which set the search start time.
      const instance = await WebAssembly.instantiate(msg.module, {
        env: {
          memory: msg.memory,
          chez_now_ms: () => performance.timeOrigin + performance.now(),
        },
      });
      exports = instance.exports;
      self.postMessage({ type: "ready" });
      break;
    }
    case "run": {
      // Give this instance its own shadow stack inside the shared memory before
      // running any deep (stack-using) code.
      exports.__stack_pointer.value = msg.stackTop;
      exports.wasm_smp_run_thread(msg.threadId);
      self.postMessage({ type: "done", threadId: msg.threadId });
      break;
    }
  }
};
