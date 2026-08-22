import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(websiteRoot, "..");

const assets = [
  [
    "assets/favicon/mrnabench-favicon-hex-black.svg",
    "public/brand/favicon-light.svg",
  ],
  [
    "assets/favicon/mrnabench-favicon-hex-white.svg",
    "public/brand/favicon-dark.svg",
  ],
  [
    "assets/mark/mrnabench-5bar-black.svg",
    "public/brand/mark-light.svg",
  ],
  [
    "assets/mark/mrnabench-5bar-white.svg",
    "public/brand/mark-dark.svg",
  ],
];

await Promise.all(
  assets.map(async ([source, destination]) => {
    const output = resolve(websiteRoot, destination);
    await mkdir(dirname(output), { recursive: true });
    await copyFile(resolve(repositoryRoot, source), output);
  }),
);
