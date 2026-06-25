import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  build: {
    lib: {
      entry: {
        index: fileURLToPath(new URL("./src/index.ts", import.meta.url)),
        "player/KumaPlayer": fileURLToPath(new URL("./src/player/KumaPlayer.ts", import.meta.url)),
      },
      formats: ["es"],
    },
    rollupOptions: {
      external: ["fflate"],
    },
    outDir: "dist",
  },
});
