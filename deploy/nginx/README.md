# Nginx proxy for the Windows MDD demo

The rendered configuration exposes the demo on the selected LAN IPv4 address
and `127.0.0.1`, with the Python backend listening only on `127.0.0.1`.
The current deployment uses `10.10.63.214:80` and backend port `8765`.
All HTTP paths go through the backend, including JavaScript, CSS, samples, and
voices. There is no Nginx filesystem root or alias for this repository.

Use the project deployment scripts to render and validate
`nginx.conf.template`; do not run the unresolved template directly. The renderer
must reject unresolved placeholders and values containing quotes, newlines,
semicolons, or Nginx variables. Validate the LAN address and ports before replacing
tokens. Use an absolute runtime directory inside this workspace.

| Placeholder | Value |
| --- | --- |
| `@@LAN_ADDRESS@@` | One explicitly selected RFC1918 IPv4 address, such as `10.10.63.214`. |
| `@@LAN_ADDRESS_REGEX@@` | The same address with every dot replaced by `[.]`, such as `10[.]10[.]63[.]214`. |
| `@@PUBLIC_PORT@@` | Public listener port, normally `80`. |
| `@@AUTHORITY_PORT_REGEX@@` | `(?::80)?` for public port 80; otherwise `:<PUBLIC_PORT>`, such as `:8080`. |
| `@@BACKEND_PORT@@` | Loopback backend port, normally `8765`. |
| `@@RUNTIME_DIRECTORY@@` | Absolute directory using forward slashes, without a trailing slash. Contains PID/logs/temp directories. |

Start Nginx with the rendered config and an explicit workspace runtime prefix
(`-p <runtime-directory>/ -c <absolute-config>`). Before `nginx -t`, create the
runtime directory, `logs`, and `temp` with its `client_body_temp`, `proxy_temp`,
`fastcgi_temp`, `uwsgi_temp`, and `scgi_temp` children. The Windows binary can
initialize those compiled default paths even when an active location overrides
its temp path. This template also explicitly places client/proxy temp paths at
`<runtime-directory>/client_body_temp` and `<runtime-directory>/proxy_temp`.
These empty directories do not enable buffering or storage.
Do not bind the Python server with `--lan` for this deployment: the proxy replaces
the accepted public Host with its exact loopback authority.

## Request boundary

Only the configured LAN address, `127.0.0.1`, and lowercase `localhost` are accepted
as incoming Host names. At port 80, an omitted or explicit `:80` is accepted;
other configured ports must appear explicitly. A nonempty Origin must use HTTP
and the **same hostname and effective port as that request's Host**. For example,
`Host: 10.10.63.214:80` with `Origin: http://10.10.63.214` is valid, while an Origin
for `localhost` is not valid for that LAN Host. Foreign hosts, `Origin: null`,
different schemes/ports, and combined duplicate Origin values are rejected.

Only after those checks does the proxy send `Host: 127.0.0.1:<backend-port>` and,
when supplied, `Origin: http://127.0.0.1:<backend-port>`. An absent or empty-valued
Origin is omitted upstream; native Nginx `$http_origin` represents both as an
empty value. `Sec-Fetch-Site` is preserved so the existing backend still rejects
`cross-site`. No CORS allowlist or wildcard CORS headers are added.

The JSON request limit is **16 MiB**, matching `MAX_BODY_BYTES` in the backend;
the backend also enforces its 12 MiB decoded-audio and 4 KiB feedback limits.
Request/response buffering and proxy caching are disabled. Requests carrying
`Transfer-Encoding` are rejected with 400 before reaching the backend; browser
JSON uploads with `Content-Length` are forwarded unchanged. The proxy never
retries a failed upstream request, including paid LLM/TTS POSTs. The 1500-second
read timeout allows long provider calls; it does not change provider timeouts or
cause automatic retries.

The public endpoint is HTTP. Remote clients can upload audio, use samples, and
analyze normally; browser microphone capture requires HTTPS. The existing UI
shows that requirement. This configuration does not install certificates.

## Validation

`tests/test_nginx_deployment.py` uses an installed Nginx binary and a local guarded
stub backend on high ports. It never calls MDD inference, LLM, or TTS providers.
Set `MDD_NGINX_EXECUTABLE` to select a binary explicitly, or let the test discover
`.tools/nginx/nginx-*/nginx.exe`. The test process owns and stops its own Nginx
instance and leaves the deployed port-80 server alone.

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_nginx_deployment.py -v
```

Relevant official documentation: [case-sensitive map expressions](https://nginx.org/en/docs/http/ngx_http_map_module.html),
[proxy headers, buffering, retries and timeouts](https://nginx.org/en/docs/http/ngx_http_proxy_module.html),
and [Windows path/prefix behavior](https://nginx.org/en/docs/windows.html).

## HTTPS ngrok ingress

`ngrok.conf.template` is a separate, standalone configuration. It adds no
listeners or public host names to the LAN configuration above. Run it with its
own workspace prefix, runtime directory, and PID. Its only listener is
`127.0.0.1:8081` by default; ngrok forwards to that loopback port and terminates
public HTTPS. The Python backend remains `127.0.0.1:8765`.

Keep the ingress closed until the running ngrok agent reports its exact HTTPS
public URL through its loopback control API. Validate that URL, render the one
hostname, run `nginx -t`, and then start the separate Nginx instance. There is no
temporary wildcard allowlist. Keep the ngrok control/inspection API loopback-only
and outside this proxy. No ngrok token belongs in either template or its output.

| Placeholder | Value |
| --- | --- |
| `@@PUBLIC_HOST@@` | The exact lowercase ASCII DNS hostname from the verified HTTPS public URL, without scheme, port, path, or wildcard. |
| `@@PUBLIC_HOST_REGEX@@` | That same validated hostname with every dot replaced by `[.]`. |
| `@@INGRESS_PORT@@` | Dedicated loopback ingress port, normally `8081`. |
| `@@BACKEND_PORT@@` | Existing loopback backend port, normally `8765`. |
| `@@RUNTIME_DIRECTORY@@` | Separate absolute workspace runtime directory, forward slashes, no trailing slash. |

The ngrok agent must preserve the public Host: use `host_header: preserve` or its
CLI equivalent, and do not select `--host-header=rewrite`. The agent's
[Host-header configuration](https://github.com/ngrok/ngrok-docs/blob/main/gateway/agent/config/v2.mdx)
describes the distinction. Nginx accepts only that exact domain, optionally with
`:443`; a supplied Origin must be HTTPS for the same domain and effective port.
`http://`, other ngrok domains, local authorities, different ports, and forged
forwarding headers cannot widen the allowlist. Accepted requests are rewritten
to the backend's existing HTTP loopback Host/Origin; absent/empty Origin stays
absent and `Sec-Fetch-Site` remains subject to the backend guard.

All paths still go through the application, including voices and unknown/private
paths. Upload limits, 1500-second read timeout, no retries, no buffering, and no
proxy caching follow the LAN policy above. Use the public **HTTPS** URL in the
browser so microphone capture has a secure origin; the loopback hop does not
require a separate local certificate.

The ngrok ingress integration tests use real Nginx and a guarded local stub on
high ports. They create no tunnel and call no inference or paid providers:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ngrok_ingress.py -v
```
