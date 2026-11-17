# Server code examples

Install nndeploy with its Server dependencies and working native bindings first.
These examples use the same Server, TaskQueue and task records as workflow JSON.

Model files and directories use the Node's existing I/O declarations:

```python
self.set_io_type(nndeploy.dag.IOType.Model)
self.add_io_param("weights")
self.weights = ["models/weights.bin", "models/tokenizer"]
```

Server copies these local resources before initialization and updates the
running Node's paths. Direct code Graphs keep their original nodes and do not
need JSON export. For saved workflows, include custom fields in the Node's
existing serialization methods; the source workflow and `.server.json` remain
unchanged. JSON entries can restore the managed paths after the source files
are removed. Factories retain the loaded resource paths across Stop/Start and
Worker replacement. A new Server still needs explicit factory registration;
matching declarations can reuse saved resources when their sources are missing.
Saved files and directories are checked before reuse. See the [resource details](../../python/nndeploy/server/README.md#graph-模型与目录资源)
for supported declarations and current recovery limits.

Resource values can also use the existing `{"file_id": "file_..."}` returned by
`POST /v1/files`. Saved JSON keeps the ID; the Worker resolves and verifies it
before Graph loading. `Dir` resources accept validated `application/zip` uploads
and become local directories. Existing file references protect the source during
initialization and execution, and persisted JSON/factory resources keep their
restart dependencies protected after Stop. No separate upload protocol is needed.

## Audio Graph

```bash
pip install 'nndeploy[server-audio]'
python demo/server/audio.py --isolation process --port 8000
```

The example uses an ordinary NumPy Graph to halve the waveform amplitude.
Upload a WAV to `POST /v1/files`, then call
`POST /v1/server/audio/deploy` with
`{"inputs":{"audio":{"file_id":"<uploaded ID>"}}}`. Download the returned
`outputs.audio.file_id` from `GET /v1/files/<ID>/content`.
Inputs are resampled to 16 kHz mono; the example accepts at most ten seconds.
The output is float32 PCM WAV. Install `ffprobe` for upload validation.

There is no audio-specific endpoint or task type. Numeric `array` and `tensor`
fields use `dtype: "float32"`, `sample_rate`, and shape `[frames]` or
`[frames, channels]`. Set `frames` to `null` for variable duration; channels
must be fixed between one and eight. The declared MIME selects WAV, FLAC, MP3,
Ogg/Opus or M4A/AAC. Encoders must support the declared rate; MP3 supports at
most two channels. WAV preserves float samples; compressed output requires
samples in [-1, 1]. Lossy codecs can change both sample values and the decoded
frame count, so use a variable frame dimension for them.

`build_graph().serialize()` can also be saved as ordinary workflow JSON, with
`config()` in the optional same-name `.server.json`. Place the node plugin in
the existing `resources/plugin` directory for the editor, or `<data_dir>/plugin`
for API-only Server. Resident Workers load those trusted plugins before Graph
initialization, including isolated processes. Workflow JSON is not rewritten.
Restart Workers to load new plugins; replacing an already imported Python
plugin in a threaded process requires restarting that process.

## Resident model budgets

```bash
python demo/server/device_limits.py --port 8000
curl http://127.0.0.1:8000/v1/server/uppercase/deploy \
  -H 'Content-Type: application/json' -d '{"inputs":{"text":"nndeploy"}}'
```

This CPU example runs independent `uppercase` and `lowercase` endpoints, leaving
one model slot for Reload. `Server(device_limits=...)` limits resident Worker
instances and declared model memory. Each entry declares its complete device
group and per-Worker memory through its existing execution configuration. These
figures are illustrative admission budgets, not limits on Python or hardware.
The example allows 256 MiB total, with 64 MiB resident memory per Worker and
16 MiB temporary memory per request through `execution.temporary_memory_bytes`.
They do not move models or set their actual devices. The example also allows two
concurrent Worker initializations or Deploy calls, reserving one slot for `lowercase`. Unused reserved
capacity cannot be borrowed by another endpoint; a dynamic batch uses one slot
for its actual algorithm call.

Resources are reserved before factories, Graph loading and warmup. Reload counts
both old and new instances until old Workers finish; insufficient capacity keeps
the old endpoint available. Shared host budgets require an explicit host identity,
stable device identities, matching policies and the same database. Controller
death or lease expiry does not free resources while an isolated child still runs.

Set `NNDEPLOY_ADMIN_TOKEN` before starting the example to inspect allocations at
`GET /api/server?include_resources=true` with the management Bearer token. The
default scope is the persistent Server workspace. Different workspaces are
independent unless explicitly configured for host coordination. This capability
covers resident models and Deploy execution in Server entries. Device waits use
the existing task deadlines and bounded queues. Optional `execution.priority`
(-10 to 10) advances queue order by one second per point, so waiting age prevents
new high-priority traffic from indefinitely overtaking older requests. Disjoint
devices proceed independently. Worker initialization and warmup also acquire an execution slot before calling
the factory. Device wait, initialization and all warmup samples share the
existing per-Worker startup deadline. Failed initialization retains its slot
until physical cleanup is confirmed; no user Task is created for startup.

Temporary memory shares the device limit with all resident Workers and running
calls. Initialization and batch input preparation reserve one request's budget;
actual batches reserve it for every valid member. Model admission also leaves
room for the configured maximum batch, including each reserved execution slot.
An impossible maximum batch is rejected before loading; batches are not resized
automatically. Cancellation does not free memory while computation still runs.
Management responses expose resident, temporary and total declared memory, plus
available bytes and required runtime headroom. Omitting temporary memory leaves
that usage unaccounted for; these estimates do not enforce allocator limits.
Use `python demo/server/device_limits.py --cpu-threads 2` to run both endpoints
in separate processes with `execution.cpu_threads=2`. This caps supported native
compute pools per Worker, including initialization, warmup and rebuilds; it is
not a process-wide OS thread quota or CPU affinity setting. Thread isolation
rejects the option because some native settings affect the whole process.
OpenCV GCD falls back to sequential execution when it cannot honor the limit.
Graph backend thread parameters are capped before init without rewriting JSON.
Unknown libraries and custom threads need algorithm-specific configuration;
PyTorch and BLAS/OpenMP performance still require backend-specific validation.
Legacy editor Run and real GPU validation remain outstanding.

## Files, audio and video through Graph ports

```bash
python demo/server/media.py --port 8000 --isolation process
# To accept MP4 containers instead:
python demo/server/media.py --port 8000 --media-type video/mp4
```

This example builds an ordinary Graph with byte ports. Upload a WAV file, then
use the returned file_id in Deploy:

```bash
curl http://127.0.0.1:8000/v1/files -F 'file=@sample.wav;type=audio/wav'
curl http://127.0.0.1:8000/v1/server/media/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"file":{"file_id":"<file_id>"}}}'
```

Download the result at `/v1/files/<output_file_id>/content`. Each request returns
a separate managed file. Server validates the MIME type and actual container;
audio/video validation requires ffprobe. The Graph receives and returns bytes,
so this example copies a complete file; it does not demonstrate audio decoding,
video frame processing or a model-specific streaming protocol.

Use `type: "image"` on PIL or OpenCV/NumPy image ports for actual image decoding
and encoding. OpenCV values use BGR/BGRA or grayscale, with uint8 image outputs.
File input ports declared as str receive the resolved local file path; output
file ports must return bytes. An optional media_type such as image/jpeg selects
the encoded output format. Workflow JSON remains unchanged.

## Native video Graph

```bash
python demo/server/video.py --port 8000 --isolation process --width 320 --height 240
# Select an encoder available in your native OpenCV build, for example:
python demo/server/video.py --port 8000 --fourcc mp4v
```

This builds `OpenCvVideoDecode → Resize → OpenCvVideoEncode`. Upload a video,
then call the same Deploy API:

```bash
curl http://127.0.0.1:8000/v1/files -F 'file=@sample.mp4;type=video/mp4'
curl http://127.0.0.1:8000/v1/server/video/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"video":{"file_id":"<file_id>"}}}'
```

Download `/v1/files/<outputs.video.file_id>/content`. Every decoded frame passes
through Resize; replace that node with your algorithm. Model nodes stay initialized
between requests. Each request closes its codec handles and returns a separate
file. Existing workflow JSON with these native codec nodes can be loaded directly;
Server detects their input/output bindings when configuration is omitted.

The default encoder is avc1. Encoder availability depends on the native OpenCV
build, which may differ from the Python cv2 package. The supported pipeline has
one video decoder and video encoder outputs, uses a sequential Graph, and accepts
at most 18000 frames and 240 fps, within the standard file/media limits. Output
keeps the input frame rate; all frames in one output must have the same dimensions.
Only the processed video track is encoded; audio tracks are not copied or mixed.
This example requires OpenCV and ffprobe, and does not require a GPU model.

## Numeric arrays and native Tensor Graphs

```bash
python demo/server/arrays.py --port 8000
# Or construct the same Graph in an isolated Worker:
python demo/server/arrays.py --port 8000 --isolation process
```

The example builds an ordinary Graph in Python, declares native Tensor ports,
warms it, and adds one to every input value. No workflow export is required:

```bash
curl http://127.0.0.1:8000/v1/server/numeric/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"values":{"dtype":"float32","shape":[2,2],"data":[1,2,3,4]}}}'
```

The output has the same dtype and shape, with flat data `[2,3,4,5]`. JSON, tasks,
events and idempotent replay always use `{dtype, shape, data}`; native objects
stay inside the Worker. Each request receives separate input storage, and outputs
are detached before another request can reuse the algorithm's buffer.

Declare `type: "array"` for NumPy code inputs/outputs, or `type: "tensor"` for
native `nndeploy.device.Tensor`. Both require `dtype` and `shape`; a `null`
dimension allows a variable size. Scalars use `shape: []` with one data element;
zero dimensions require empty data. `data` is a flat, row-major list, not nested
lists. Integer dtypes reject floats and overflow; floating dtypes require finite,
representable values and round to the declared precision. Optional `minimum` and
`maximum` apply to each element. Clients must preserve JSON integer precision
when using 64-bit integer values; binary array-file transport is not yet provided.

Each field defaults to at most 1,048,576 elements and 8 MiB of numeric storage.
Configure `max_elements` and `max_bytes` explicitly as needed, up to 16,777,216
elements / 64 MiB per field. The HTTP JSON body limit still applies separately;
JSON's encoded size exceeds the numeric storage size. Shape, data length, dtype,
value bounds and byte limits are checked before native allocation. Input models,
warmup, synchronous/async Deploy, streams and dynamic batches use these checks.
Output dtypes are checked rather than silently cast. MySQL's task payload columns
are migrated to LONGTEXT so valid requests/results can exceed 64 KiB.

Code array fields support boolean, signed/unsigned 8/16/32/64-bit integers,
float16/32/64 and up to eight dimensions. Native Tensor fields use the numeric
dtypes and support up to five dimensions, matching the existing native converter.
Tensor inputs default to CPU; a trusted input-field `device` such as `"cuda:0"`
selects a compiled, available backend. Requests cannot override it. Output Tensor
shape, dtype and storage size are checked before copying device data to host.
CPU paths are tested locally; actual GPU transfers await the hardware environment.
Without a `data_format` declaration, the native converter continues to infer
format from rank. A Tensor field can explicitly declare `data_format` using the
existing native names: N, NC, NCL, S1D, NCHW, NHWC, OIHW, NCDHW, NDHWC or NDCHW.
Its configured shape describes the Tensor received by the algorithm. S1D requires
a fixed middle dimension of 1. Packed NC4HW/NC8HW and Auto are not accepted.

With this declaration, requests may include `data_format` to describe their
source data. Omitting it uses the field's format. Server checks the shape after
permutation, validates values and limits, then transposes into the declared
format before creating the Tensor. It only converts formats with the same named
axes: NCHW ↔ NHWC and permutations of NCDHW/NDHWC/NDCHW. It never relabels OIHW
weights as NCHW activations. Tensor outputs use their actual native format and
are converted to the declared output format before persistence. Results include
`data_format`, and their flat `data` follows that result's shape in row-major order.
Requests cannot override the configured destination format or target device.
Legacy fields without a format declaration retain their existing schema and
three-key dtype/shape/data wire format.

Try the four-dimensional example with an NHWC Graph and NCHW request:

```bash
python demo/server/arrays.py --isolation process --data-format NHWC
curl http://127.0.0.1:8000/v1/server/numeric/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"values":{"dtype":"float32","shape":[1,2,2,3],"data_format":"NCHW","data":[0,1,2,3,4,5,6,7,8,9,10,11]}}}'
```

The result has shape `[1,2,3,2]`, format NHWC and data
`[1,7,2,8,3,9,4,10,5,11,6,12]`. This performs axis conversion and adds one to
each value. Algorithms creating new Tensors must set their actual format rather
than relying on rank inference when that inference is incorrect.

Graph numeric fields bind exposed `edge` names and require matching declared
native port types. Python ports declared as `numpy.ndarray` support the same
bounded scalar through eight-dimensional numeric arrays as code handlers,
including float16, int64/uint64, bool, empty dimensions and HWC singleton channels.
Arrays retain their dtype, shape and Python owner in the Edge; read-only and
strided arrays are not silently converted into images. Server detaches outputs
before storing task results, so later requests cannot overwrite saved values.

An edge connected to a native `cv::Mat` port continues to use OpenCV conversion.
Such `array` bindings require 2D or HWC arrays with
uint8/int8/uint16/int16/int32/float32/float64; HWC requires 2–512 fixed channels,
and single-channel arrays use a 2D shape. Incompatible native image declarations
fail at startup. Untyped standalone Edge submissions retain the previous image
behavior; use `edge.set_type(np.ndarray)` or the existing Node input/output type
declarations for general arrays. This requires rebuilding the Python bindings.

Try an ordinary one-dimensional NumPy Graph in an isolated Worker:

```bash
python demo/server/arrays.py --isolation process --value-type array
curl http://127.0.0.1:8000/v1/server/numeric/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"values":{"dtype":"float32","shape":[4],"data":[1,2,3,4]}}}'
```

The result is `[2,3,4,5]` with shape `[4]`. The same Graph can accept a different
vector length on the next request; the default Tensor example is unchanged.

For binary arrays, set the existing field `media_type` to `application/x-npy`
on each input or output that should use files. The field remains `array` or
`tensor`, with the same dtype, shape and capacity limits. Upload with
`POST /v1/files`, pass `{"file_id":"..."}` and download output from
`GET /v1/files/{file_id}/content`. This uses the same permissions, task references,
retention and object storage as other files; it introduces no separate upload
or deployment flow. Fields without this MIME type continue to use inline JSON.

```bash
python demo/server/arrays.py --isolation process --value-type array --npy
python -c 'import numpy as np; np.save("input.npy", np.array([1,2,3,4], dtype="float32"))'
curl http://127.0.0.1:8000/v1/files -F 'file=@input.npy;type=application/x-npy'
# Use file_id from the upload response:
curl http://127.0.0.1:8000/v1/server/numeric/deploy \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"values":{"file_id":"<uploaded file ID>"}}}'
# Use file_id from outputs.values:
curl 'http://127.0.0.1:8000/v1/files/<output-file-ID>/content' -o output.npy
python -c 'import numpy as np; print(np.load("output.npy", allow_pickle=False))'
```

The output is `[2, 3, 4, 5]`. NPY v1/v2 plain numeric arrays support both byte
orders and C/Fortran layout. Workers receive independent native-endian arrays;
changing an algorithm input does not modify the uploaded file. Upload rejects
pickle/object/structured/complex data, archives, nonfinite numbers, oversized
headers, truncated content and trailing bytes. Field shape, dtype, value bounds
and capacity are checked before task admission and again in the Worker.
`max_bytes` measures numeric data; the storage upload limit also includes the NPY
header. A Tensor file uses its field's declared `data_format`, since NPY itself
does not carry semantic axis names; outputs are converted to the configured
format before saving. File results and stream events retain independent files
before an algorithm can reuse its buffer. The `--npy` example initializes the
Graph without a warmup request; deployments can configure warmup using uploaded
file IDs in the ordinary `execution.warmup` samples.

## Warmup before Ready

Add samples to the existing configuration; each initialized Worker runs them
before accepting requests. For a text workflow with an input named `text`:

```json
{
  "execution": {
    "startup_timeout_seconds": 60.0,
    "warmup": [{"inputs": {"text": "warmup"}, "parameters": {}}]
  }
}
```

Merge this with the workflow's existing input/output bindings in
`workflow.server.json`. The workflow JSON itself stays unchanged. Python code
uses the same setting:

```python
from nndeploy.server import Server

server = Server(
    name="echo",
    deploy=lambda inputs: inputs,
    config={
        "inputs": {"text": {"type": "string"}},
        "outputs": {"text": {"type": "string"}},
        "execution": {"warmup": [{"inputs": {"text": "warmup"}}]},
    },
)
server.run()
```

Samples use normal input/default/parameter and output validation. Typed Python
inputs use the supplied input model. The default is an empty list; at most 16
samples of 1 MiB each are accepted. Use existing `file_id` references for images.
Saved and running configurations retain those input files for later restarts.

Warmup executes real algorithm code and repeats when a Worker is recreated.
Choose samples whose algorithm-specific side effects are appropriate for startup;
external writes by custom code or nodes are not rolled back. Server does not
create task or event history for samples. Managed Graph file outputs are validated
inside the request's temporary directory and discarded without upload or file
metadata. Process termination can leave temporary directories for later recovery.

Serial, concurrent, async engine and streaming handlers use their actual execution
paths. Dynamic batching warms each sample as a one-item batch through `batch_key`
and `deploy_batch`; custom model-specific batch shapes can be initialized in the
handler's existing `init()` method. Samples run sequentially per Worker.

The startup deadline includes initialization and all samples. Process isolation
can enforce it by confirming process exit. Threads cannot interrupt native calls:
they wait for computation to return, then report an exceeded startup deadline and
release the Worker. Async initialization observes cancellation and waits for
cleanup; blocking native calls still need process isolation. A failed warmup never
becomes Ready. Reload warms the candidate before switching requests; failure
preserves the previous running and saved configuration.

## Task retention

Keep completed tasks for one day by adding this to the existing entry configuration:

```json
{
  "execution": {
    "task_retention_seconds": 86400.0
  }
}
```

The default is `null` (keep records); the minimum enabled duration is 60 seconds.
Each task saves its policy when accepted. The database clock starts retention at
terminal commit, so queued or running tasks are never removed. Task queries expose
`purge_after`. Background cleanup deletes an expired task, its events, idempotency
key and task file references in one transaction. File content has its own retention
policy. After cleanup, querying the task returns 404 and reusing
its idempotency key may execute a new task.

`await server.cleanup_tasks()` runs one bounded batch immediately. The equivalent
HTTP call is `POST /api/server/cleanup-tasks` with the management token. Cleanup is
safe to run from another Server sharing the same database; each task retains the
original policy.

To also expire unreferenced files, pass `storage_config={"retention_seconds": 86400.0}`
to `Server` (or include it in the CLI storage JSON). This optional duration starts
when the file metadata is created, with a minimum of 60 seconds. Tasks, saved
workflows, live runtimes and active readers keep their files protected even after
that time. Signed download URLs delay automatic cleanup until their expiry.
Explicit authorized deletion can invalidate a signed URL earlier.

`await server.cleanup_files()` or management-only `POST /api/server/cleanup-files`
runs one bounded batch and reports scanned, deleted, skipped and failed counts.
Failed deletions remain recorded and retry after a delay. Failed task outputs are
eligible for cleanup; successful outputs stay referenced for the task's lifetime.
For programmatic reads, `with server.storage.open(file_id) as content:` holds the
file until the stream closes. Files in other local storage directories are isolated.
Legacy records without a verified storage location and crashed runtimes' references
are retained pending reconciliation.

## Incremental output

```bash
python demo/server/stream.py --port 8000
```

```bash
curl -N http://127.0.0.1:8000/v1/server/words/deploy \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"mode":"stream","inputs":{"text":"one two three"}}'
```

The response contains numbered `start`, `delta`, `result` (or `error`) and `done`
events. The result/error event and `done` are committed with the task's terminal
state and contain a status URL. Read that URL for final outputs or an error. The `X-Task-ID` response
header also identifies the task. A delta event contains `task_id`, `sequence` and
`data` (the yielded JSON value). The last yielded value becomes the final output
and is checked against the entry's output schema; intermediate values are bounded
JSON events. Generator return values are not used.

The same generator supports `mode: "sync"` (wait for its final result) and
`mode: "async"` (return its task ID immediately). Generator support must be
declared with `streaming.enabled`. Both Python generators and async generators
are supported. Graph entries currently expose final outputs only.

Disconnecting does not cancel the task. Resume on the same Server, or on another
instance sharing the task database:

```bash
curl -N http://127.0.0.1:8000/v1/tasks/TASK_ID/events \
  -H 'Accept: text/event-stream' -H 'Last-Event-ID: 2'
```

Only events after the supplied sequence are replayed. Reconnecting again may
repeat an event that arrived before the client saved its sequence; clients should
deduplicate by task ID and sequence. Task ownership and entry permissions apply
to event readers. This local example omits API keys; applications can supply the
same `auth_config` used by other Server entries.

Use the existing task cancel endpoint to stop iteration. Threads check cancellation
at output boundaries and do not free their execution slot while blocked native
code is still running. Process isolation can terminate the process and confirm
its exit. Async generator iteration also has a cooperative timeout.

Defaults: 10,000 incremental outputs, 64 KiB per yielded JSON value and 300 seconds of stream
execution, adjustable in `streaming`. The SSE reader fetches at most 16 events at
a time; each HTTP connection lasts up to 60 seconds and each socket send has a
5-second deadline. Reconnect to continue longer tasks. Concurrent event connections
are bounded by the Server's `max_inflight_tasks` setting. DB failures close readers
with an unnumbered retryable `error` event (`terminal: false`); they do not manufacture a successful terminal
state. A failed event write stops the producing request.

This implementation persists events in the configured SQL database; it does not
claim token-generation throughput or event-retention cleanup is complete.

## Concurrent async engines

The same example can use one async engine with four request slots:

```bash
python demo/server/stream.py --engine-concurrency 4
```

Requests use the same Deploy and task endpoints. They can overlap in one engine
without loading four separate model instances. In application code, set
`execution.scheduling` to `engine` and `execution.max_concurrency_per_worker` to
the required bound. `workers` still controls how many independent engine objects
are created. Each object's factory, initialization, async Deploy calls and cleanup
run on its own event-loop thread. Graph instances retain serial execution.

Engine entries require `async def` functions or async generators. A synchronous
blocking call inside an async function will still block that engine's loop;
the adapter must use the algorithm's asynchronous submission interface. These
examples process text and validate transport/scheduling; they do not benchmark
an LLM backend or GPU.

Cancellation defaults to waiting for actual engine computation to finish. An
algorithm may declare `execution.cooperative_cancel: true` only when cancellation
of its coroutine also waits for the underlying computation and resource cleanup
to end. That declaration permits per-request cooperative execution deadlines.
Cancellation and cleanup for one request do not cancel other requests. If cleanup
is still running, its slot remains occupied. Without this declaration, a cancelled
request or failed event write suppresses further output but drains the computation
before recording the terminal state.

Algorithms can read their existing task identity with
`nndeploy.server.logging_taskid.get_task_id()`; it is isolated per concurrent
request. Hard termination of a blocking native engine requires process isolation;
this concurrent engine implementation currently uses threads and rejects process
isolation configuration explicitly.

## Explicit batch calls

The running words example also accepts a bounded list of inputs:

```bash
curl http://127.0.0.1:8000/v1/server/words/deploy/batch \
  -H 'Content-Type: application/json' \
  -d '{"mode":"sync","items":[{"inputs":{"text":"one two"},"idempotency_key":"example-1"},{"inputs":{"text":"three four"},"idempotency_key":"example-2"}]}'
```

The response's `items` preserve input order and contain `index`, `accepted`,
`status_code`, and either a task ID with outputs or an error. HTTP 200 does not
mean every item succeeded: inspect each item's status. Async mode returns 202
when at least one item has a task record; query the usual task endpoints.

Requests allow 1–64 items, all for the named entry. Each accepted item is an
ordinary Task with the same validation, ownership, limits and cancellation rules.
Malformed request envelopes are rejected before admission. Individual invalid
inputs, exhausted capacity and algorithm failures are reported per item; accepted
items are not rolled back. A database admission failure prevents further writes
in that batch. A concurrent Reload rejects unaccepted items rather than silently
running them with different code or configuration.

Retry using each item's original `idempotency_key`. Already recorded items retain
their task IDs and final errors/results; previously rejected items can be accepted
when capacity returns. The batch endpoint rejects a top-level `Idempotency-Key`
header. It has no separate batch ID or batch lifecycle. In Python, use
`await server.deploy_batch(name, items, mode="sync")`.

Explicit batch calls submit independent tasks. Dynamic algorithm batching is
configured separately for handlers that declare compatible batching support.

## Dynamic algorithm batching

Run a CPU-only example that combines vectors in one NumPy operation:

```bash
python demo/server/batch.py --batch-size 8 --batch-wait-ms 20
```

Send concurrent requests to `POST /v1/server/vectors/deploy` with
`{"inputs":{"values":[1.0,2.0]}}`. Each response retains its own `task_id`; its
outputs contain the sum and the actual `batch_size`. The explicit `/deploy/batch`
endpoint can also submit items to the same queue. A lone request runs after the
collection window without waiting for a full batch.

Enable batching through the existing entry configuration:

```python
config = {
    "execution": {
        "batching": {"enabled": True, "max_size": 8, "max_wait_ms": 20},
        "max_queue_size": 32,
    }
}
```

The handler provides two synchronous methods:

- `batch_key(inputs, **parameters)` returns a small JSON value describing
  compatible input shapes/types. Choose a key that is safe for your algorithm;
  Server additionally separates callers and complete parameter values.
- `deploy_batch(items)` receives a list of `task_id`, `request_id`, `inputs`, and `parameters`
  dictionaries. Return one output per item in the same order. An individual
  `ServerError` in this list fails only that task. Raising an exception or
  returning a different number of outputs fails all participating tasks.

Each Worker owns one handler and one thread for initialization, compatibility
keys, batch calls, and cleanup. It reserves at most `max_size` request slots;
entry capacity is `max_queue_size + workers * max_size`. The oldest compatible
group goes first, and later arrivals do not extend its collection window.
`max_wait_ms` limits collection delay, not time spent behind a busy algorithm.
Queue deadlines include batch collection and are rechecked before computing.

`request_id` identifies the original HTTP request and is `None` for direct Python
calls or older tasks. A batch has no single request context; use each item's
explicit ID for correlation. `batch_key` runs with that item's request/task context.

Cancellation before execution skips that item. Once a batch is computing,
cancelling one task waits for the computation to finish, retaining its slot and
letting the other items finish. Reload initializes a new handler, routes new
requests to it, and drains the old handler's batches.

This implementation requires serial thread workers and non-streaming results.
Ordinary Graph instances, process workers and engine scheduling reject external
batching. Engines with their own batching continue to use `scheduling="engine"`.
Real model/GPU throughput remains a separate acceptance test.

## Algorithm-owned HTTP protocols

Run the CPU text example:

```bash
python demo/server/algorithm.py --port 8000
curl http://127.0.0.1:8000/v1/server/text/text \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: greeting' \
  -d '{"text":"Hello"}'
```

The synchronous response is `{"id":"task_...","text":"HELLO"}`. Setting
`"mode":"async"` returns the normal 202 response and `/v1/tasks/{task_id}` URL.
The standard `/v1/server/text/deploy` endpoint remains available. Both routes
share the same queue, identity, file permissions, task records and idempotency
scope. A custom protocol changes the HTTP shape; it does not create a new model
instance or run the algorithm in the HTTP handler.

The example keeps `protocol.py`, `executor.py`, `api.py` and `create_entry()` in
`text/server/`. Real algorithms can use the same layout under their own directory:

```python
server = Server(data_dir="./resources/server")
server.register("text", "text.server:create_entry")  # No import yet.
server.add(name="text", algorithm="text", options={"uppercase": True})
```

Registration is local to this Server and trusted startup code. Only selecting an
algorithm imports its factory; missing optional dependencies report an error for
that selection. HTTP cannot register factories or import modules. The factory
returns a dictionary of existing `Server.add()` arguments: exactly one execution
source, optional config/input_model/output_model and an optional FastAPI router.
It prepares configuration; model initialization belongs in the Graph factory or
handler lifecycle, on the existing Worker. Explicit `config` or models passed to
`add()` replace factory values. Restart the same application bootstrap to restore
code registrations and custom routes; Python callables are not stored in the DB.

Registration is optional. A direct Graph, workflow or code source can simply use
`server.add(name="text", deploy=deploy, router=router)`.

Algorithm routes use `Depends(get_deploy)` from `nndeploy.server.dependencies`:
translate the body to `DeployRequest`, await the dependency, and optionally map
the synchronous task result to the algorithm's response. Async and stream modes
return HTTP Response objects directly. This preserves `Authorization`,
`Idempotency-Key`, `Accept`, task ownership and stream limits. A streaming entry
must declare streaming and the caller also needs `tasks:read` permission.
Keep request conversion small; expensive preprocessing belongs in the Worker.

Server mounts the router under `/v1/server/{name}` and enforces `deploy`
permission for that entry on every route, including auxiliary routes. Routes must
have a literal first segment other than `deploy` or `schema`. Add routes before
Server starts; currently use a plain HTTP APIRouter without nested router lifespan
composition, startup/shutdown hooks or WebSocket routes. Entry Start/Stop/Reload
continue to control the model through the existing lifecycle.

Other dependencies expose the current entry description (`get_entry`) and
shared DB/Storage (`get_db`, `get_storage`). `get_redis` returns the current
entry's asynchronous Redis client when Redis scheduling is active, otherwise 503.
Algorithm-specific records can use the shared DB, and must enforce their own
record ownership. Never close these shared dependencies from algorithm code.
For Redis or process workers, execution factories still need to be importable;
return top-level functions and put varying defaults in config, rather than
capturing them in closures.

## Thread-safe synchronous code

For a synchronous handler explicitly designed to be called from multiple threads,
configure `execution.scheduling="concurrent"` and
`execution.max_concurrency_per_worker=2`. One handler is initialized per Worker;
its requests run on a bounded thread pool. Inputs and task contexts are isolated,
but protecting mutable shared model state is the handler's responsibility.
Graph instances retain serial execution; async handlers use engine scheduling.
Cancellation waits for synchronous computation to finish and does not release a
slot while a native thread remains active.

### Duplex input and output

Run `python demo/server/duplex.py --port 8000`. A duplex request is still one Task. Create it with `mode: "async"`, then send ordered input using HTTP or connect to the Task's WebSocket:

```python
import json
import requests
from websockets.sync.client import connect

base = 'http://127.0.0.1:8000'
response = requests.post(base + '/v1/server/text/deploy', json={
    'mode': 'async', 'inputs': {'prefix': 'Result: '}
})
response.raise_for_status()
task_id = response.json()['task_id']
with connect(f'ws://127.0.0.1:8000/v1/tasks/{task_id}/stream') as socket:
    print(json.loads(socket.recv()))  # ready
    socket.send(json.dumps({'sequence': 1, 'data': {'text': 'hello '}}))
    socket.send(json.dumps({'sequence': 2, 'data': {'text': 'world'}}))
    socket.send(json.dumps({'sequence': 3, 'end': True}))
    while True:
        event = json.loads(socket.recv())
        print(event)  # input_ack and output events may interleave
        if event['event'] == 'done':
            break
print(requests.get(base + f'/v1/tasks/{task_id}').json())
```

Each `sequence` starts at 1 and increases by one. `input_ack` confirms durable acceptance, not completion of computation. If an acknowledgement is lost, retry the same sequence and content; different content returns `input_conflict`. Send `end: true` without data to finish input. For HTTP, POST the same messages to `/v1/tasks/{task_id}/inputs`; output may be read through the existing SSE events endpoint. Python callers use `await server.send_input(task_id, data, sequence=1)` and `await server.send_input(task_id, sequence=2, end=True)`.

Reconnect to `/v1/tasks/{task_id}/stream?after=<last_output_sequence>` to replay output. The `ready` message includes the last accepted input sequence and whether input is closed. Retry any unconfirmed input instead of assuming it was processed. Output event sequences and input sequences are independent. Disconnecting does not cancel computation; use the existing cancel endpoint.

Algorithms declare `streaming.enabled: true` and `streaming.duplex: true`, and receive an asynchronous `input_stream` iterator as a keyword argument. Normal initial inputs and parameters keep their existing meaning. Each iterator value is bounded JSON; the algorithm validates its chunk format. File references in chunks obey the same ownership and retention rules as initial Task inputs. Audio/video wire formats and binary WebSocket frames are not automatically converted. Async generators work in serial thread/process workers and in async engine mode, including Redis shared workers. Warmup supplies an empty input stream; algorithms should produce a final result for that case if warmup is configured.

`max_pending_inputs` bounds unconsumed input (default 64). When full, HTTP returns retryable 429 and WebSocket sends a nonterminal `input_queue_full` error; retry the same sequence after the algorithm consumes earlier input. One final end marker can be appended even when full. `max_events` and `max_event_bytes` apply independently to input and output. Accepted input remains with its Task until the configured task retention cleanup, enabling idempotent retries. Migration `0030_task_inputs` adds internal input records; no workflow JSON changes are needed.

WebSocket requires the Task owner's `deploy` and `tasks:read` permissions (or administrator access), uses the normal Bearer header and same-origin/explicit CORS checks, and shares the existing Server stream connection limit with SSE. This header authentication is intended for SDK clients; browser-native WebSocket cannot supply an arbitrary Authorization header. Cookies and query-string tokens are not accepted as substitutes. Connections last at most 60 seconds, input inactivity closes them after 30 seconds, sends have a 5-second limit, and output has a 10-second heartbeat. Reconnect when needed. With audit enabled, input frames produce metadata-only audit records; denied handshakes and every read-only WebSocket action are not yet covered by complete WebSocket audit.

### MCP tools

`mcp_algorithms.py` exports two typed Python algorithms at `/mcp`, with optional
`--workflow path/to/workflow.json` to export an entire multi-node Graph as a third
tool. `mcp_client.py` discovers and calls them using the official MCP SDK, then
polls task results. Install `nndeploy[server-mcp]` first. See the Server README's
MCP section for authentication, media inputs and deployment configuration.
