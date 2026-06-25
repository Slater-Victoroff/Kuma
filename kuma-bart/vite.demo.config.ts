import { defineConfig, type Plugin } from "vite";
import { readdirSync } from "node:fs";
import { resolve } from "node:path";

// process.cwd(), not an import.meta.url-relative path: Vite bundles config files
// through a temporary esbuild step before evaluating them, which can make
// import.meta.url inside a *config* file resolve relative to that temp location
// instead of this file's real location -- a well-known Vite gotcha specific to config
// files (app source under demo/ doesn't have this problem). The dev server's process
// is always started from the package root (kuma-bart/ locally, /workspace in
// Docker -- see Dockerfile's WORKDIR), so cwd-relative is the reliable pattern here.
const ARTIFACTS_DIR = resolve(process.cwd(), "demo/artifacts");

/** Dev-server-only endpoint (GET /api/artifacts -> string[] of filenames) so the demo
 * can populate a dropdown of available .iph files instead of asking for a hand-typed
 * path. No production build counterpart needed -- this demo is never static-built and
 * deployed, only run via `vite`/`vite --host` (see kuma-bart/README.md). */
function artifactsApiPlugin(): Plugin {
  return {
    name: "kuma-bart-artifacts-api",
    configureServer(server) {
      server.middlewares.use("/api/artifacts", (_req, res) => {
        let files: string[] = [];
        try {
          files = readdirSync(ARTIFACTS_DIR)
            .filter((name) => name.endsWith(".iph"))
            .sort();
        } catch (err) {
          // demo/artifacts doesn't exist (e.g. no bind mount outside Docker) -- empty
          // list for the browser, but logged here so a real misconfiguration (wrong
          // path, permissions) doesn't silently look identical to "just no files yet."
          console.error(`[kuma-bart-artifacts-api] couldn't read ${ARTIFACTS_DIR}:`, err);
          files = [];
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(files));
      });
    },
  };
}

export default defineConfig({
  // demo/artifacts is bind-mounted from the repo root's artifacts/ by docker-compose's
  // bart-demo service, so it's servable directly at /artifacts/... with no fs.allow
  // overrides needed.
  root: "demo",
  plugins: [artifactsApiPlugin()],
});
