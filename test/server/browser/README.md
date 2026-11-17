# Editor browser recovery checks

`run_reconnect.py` starts a disposable loopback Server, saves an image workflow,
and runs real headless Chrome against the local frontend build. It closes the
Server on success or failure and retains screenshots, request identities and
actual Graph execution counts in a fresh output directory.

Prerequisites: the Python/native nndeploy development environment, a built
`app/workflow/dist`, Node.js, Chrome and an installed Playwright module.

```bash
python test/server/browser/run_reconnect.py \
  --playwright /absolute/path/to/node_modules/playwright \
  --chrome /absolute/path/to/chrome
```

The test checks that a real WebSocket disconnect reconnects without submitting
another task; an accepted request whose response is lost survives a page reload;
an explicit Retry preserves both task and workflow identities; two intentional
runs execute exactly twice despite three HTTP submissions; and saved image
history still renders. A third run from Webui verifies that cancellation shows
Stopping until execution actually ends, then returns to Run without saved files.
The offline fixture deliberately omits the template
directory, exercising the Server's `message` error format; notifications must
contain readable text, and uncaught browser errors fail the check.

Pass `--scenario nested` to check two nested Graphs whose output nodes share the
same name. Their distinct image pixels must render correctly on the canvas,
in Webui, and in history after deleting both original files and reloading.
It also checks that the loaded Graph fits inside the editor, that explicit Auto
Layout separates the expanded previews, and that Save/reload preserves node
positions. The saved Graph must still run with the correct two outputs.

`launch_reconnect.py` is a test-only process with fault injection middleware.
Never use it as a production launch entrypoint. No fault routes or flags are
added to the production Server.

Pass `--scenario resources` to verify the authenticated model resource flow:
canvas model-file and ZIP-directory selection, invalid ZIP without losing the
previous value, Save/reload, Run, Webui replacement and Run, Server Start,
HTTP Deploy and Stop. The fixture uses small real Graph resource nodes; it is
not a real model/GPU or performance acceptance test. It also holds an upload
response to verify array-item removal is disabled, and uses a literal dotted
resource field to catch form-path rewriting. Credentials are disposable local
test values; uploads use the browser session without a bearer token.

Pass `--scenario mcp` (with the optional `server-mcp` dependencies installed) to
check the Server panel's export switch, tool name/description, actual clipboard
copy, persisted fields after page reload, tool renaming and disabling. The runner
starts the disposable editor with `--mcp`; screenshots are saved with the evidence.
