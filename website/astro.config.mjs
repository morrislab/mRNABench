import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

const base = process.env.SITE_BASE ?? "/";
const baseRoot = base.endsWith("/") ? base : `${base}/`;

export default defineConfig({
  site: process.env.SITE_URL ?? "https://morrislab.github.io",
  base,
  output: "static",
  trailingSlash: "never",
  redirects: {
    "/overview": baseRoot,
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
