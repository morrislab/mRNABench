export const site = {
  name: "mRNABench",
  title: "mRNABench | A benchmark for mRNA representation learning",
  description:
    "A benchmark for evaluating frozen nucleotide-model representations on mature mRNA tasks.",
  repository: "https://github.com/morrislab/mRNABench",
  paper: "https://doi.org/10.1101/2025.07.05.662870",
};

export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, "");
  const relativePath = path.replace(/^\/+/, "");

  if (!relativePath) {
    return base || "/";
  }

  return `${base}/${relativePath}`;
}
