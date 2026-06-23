import { defineConfig } from "vite";

export default defineConfig({
  // demo/artifacts is bind-mounted from the repo root's artifacts/ by docker-compose's
  // bart-demo service, so it's servable directly at /artifacts/... with no fs.allow
  // overrides needed.
  root: "demo",
});
