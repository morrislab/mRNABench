import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

const base = process.env.SITE_BASE ?? "/";

export default defineConfig({
  site: process.env.SITE_URL ?? "https://morrislab.github.io",
  base,
  output: "static",
  trailingSlash: "never",
  redirects: {
    "/overview": "/",
  },
  integrations: [sitemap()],
  vite: {
    server: {
      allowedHosts: [".use.devtunnels.ms"],
    },
    build: {
      cssMinify: "lightningcss",
    },
  },
});
