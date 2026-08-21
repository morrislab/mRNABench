# Dataset source artifacts

## Ietswaart RNA lifecycle dataset processing

This directory contains the intermediate transcript tables used by mRNABench
to construct `rna-lifecycle-ietswaart`.

For each transcript, mRNABench averages replicate coverage in the chromatin,
cytoplasm, and polysome fractions, then normalizes those three values to sum to
one. A compartment receives a positive label when its normalized coverage
share is below that compartment's 33rd percentile across dataset rows. The
target order is `[chromatin, cytoplasm, polysome]`.

### Original direct-RNA data

Ietswaart et al. generated direct RNA sequencing data from four K562 RNA
fractions:

- chromatin-associated RNA;
- cytoplasmic RNA;
- polysome-associated RNA; and
- whole-cell (`total`) RNA.

Each fraction has two biological replicates. Whole-cell RNA is used as an
expression filter in the mRNABench processing.

Oxford Nanopore sequencing records RNA molecules as electrical signal in FAST5
files. Basecalling converts this signal into nucleotide sequences and quality
scores in FASTQ format. Metadata in the GSE208225/SRP386439 FAST5 files
records:

- Guppy `5.1.13+b292f4d13`;
- `rna_r9.4.1_70bps_fast.cfg`;
- sequencing kit `SQK-RNA002`; and
- flow cell `FLO-MIN106`.

The paper reports retaining reads with a basecalling score above 7. Project
correspondence records that the study authors supplied the complete
localization FASTQ set directly for the mRNABench reprocessing. The files were
subsequently copied to the mRNABench compute environment and used as the
inputs described below. Public study records provide the associated sequencing
accessions and FAST5 signal files.

Researchers seeking the same consolidated FASTQ files can contact the original
study authors.

### mRNABench FASTQ-to-table processing

The mRNABench reprocessing used this input layout. `barcode01` through
`barcode08` are workflow directory identifiers, not molecular barcodes.

| Directory | Fraction | Replicate | FASTQ |
|---|---|---:|---|
| `barcode01` | Cytoplasm | 1 | `K562_cyto_rep1_pass.fastq` |
| `barcode05` | Cytoplasm | 2 | `K562_cyto_rep2_pass.fastq` |
| `barcode02` | Chromatin | 1 | `K562_chr_rep1_pass.fastq` |
| `barcode06` | Chromatin | 2 | `K562_chr_rep2_pass.fastq` |
| `barcode03` | Polysome | 1 | `K562_poly_rep1_pass.fastq` |
| `barcode07` | Polysome | 2 | `K562_poly_rep2_pass.fastq` |
| `barcode04` | Total | 1 | `K562_total_rep1_pass.fastq` |
| `barcode08` | Total | 2 | `K562_total_rep2_pass.fastq` |

The workflow was `wf-transcriptomes` v1.6.1 at Git revision
`a33f3b967c797ce95262684ac4000d7897241e67`, with:

- minimap2 2.24-r1122;
- samtools 1.17;
- bedtools 2.30.0;
- seqkit 2.2.0;
- StringTie 2.1.1;
- gffcompare 0.11.2;
- gffread 0.12.7; and
- fastcat 0.10.2.

The reference inputs were the GENCODE v47 GRCh38 primary-assembly genome and
annotation:

- `GRCh38.primary_assembly.genome.fa.gz`
- `gencode.v47.annotation.gtf.gz`

The completed run used:

```bash
nextflow run epi2me-labs/wf-transcriptomes \
  -r a33f3b967c797ce95262684ac4000d7897241e67 \
  -profile singularity \
  -resume \
  --de_analysis \
  --direct_rna true \
  --fastq /path/to/barcode-directories \
  --ref_genome /path/to/GRCh38.primary_assembly.genome.fa.gz \
  --ref_annotation /path/to/gencode.v47.annotation.gtf.gz \
  --sample_sheet /path/to/sample_sheet.csv \
  --threads 4 \
  --minimap2_index_opts "-k14" \
  --minimap2_opts "-uf" \
  --minimum_mapping_quality 40 \
  --poly_context 24 \
  --max_poly_run 8 \
  --bundle_min_reads 50000 \
  --stringtie_opts "--conservative" \
  --gffcompare_opts "-R" \
  --out_dir /path/to/output
```

For workflow compatibility, the sample sheet assigned `control` to chromatin
and cytoplasm and `treated` to polysome and total. These values configure the
optional differential-expression branch; dataset construction uses only the
per-sample transcript tables.

The workflow:

1. Concatenates each directory's FASTQ input with fastcat.
2. Builds a minimap2 index with `-k14 -I 1000G`.
3. Aligns direct-RNA reads with `minimap2 -ax splice -uf`.
4. Retains primary alignments with mapping quality at least 40 using
   `samtools view -q 40 -F 2304`.
5. Filters likely internal-priming alignments using 24-nt genomic flanks and a
   poly(A) run threshold of 8.
6. Sorts alignments and splits large BAMs into bundles of at least 50,000
   reads.
7. Runs reference-guided long-read StringTie with
   `--rf -G <annotation> -L -v -p 4 --conservative`.
8. Merges each sample's transcript annotations.
9. Compares each sample annotation with GENCODE v47 using
   `gffcompare -R -r <annotation>`.
10. Parses each gffcompare `.tmap` into one
    `<alias>_transcripts_table.tsv` per sample.

Each table contains reference gene and transcript identifiers, gffcompare
class code, query transcript identifier, exon count, transcript coverage, and
transcript length.

### Transcript tables to benchmark labels

`RNALifecycleIetswaart._get_data_from_raw()` performs these steps:

1. Keep rows with `cov > 0` in each replicate. `cov` is the StringTie
   transcript coverage reported by `wf-transcriptomes`; replicate values are
   averaged directly without cross-library scaling.
2. Inner-join each replicate pair on reference gene, reference transcript,
   class code, exon count, and transcript length.
3. Retain gffcompare `class_code == "="`, indicating a matching intron chain
   between the query and reference transcript.
4. Average coverage across the two replicates for each fraction.
5. Outer-join chromatin, cytoplasm, polysome, and total coverage.
6. Set missing subcellular coverage to zero.
7. Keep rows with positive total coverage and positive coverage in at least
   one subcellular fraction.
8. Normalize chromatin, cytoplasm, and polysome coverage within each row so the
   three proportions sum to one.
9. Calculate the 33rd percentile independently for each fraction.
10. Assign a positive label when a row's normalized coverage is strictly below
    that fraction's 33rd percentile.
11. Add transcript sequence, CDS, splice, gene, and chromosome features from
    GenomeKit's GENCODE v47 annotation.

The joins intentionally preserve the source rows, including many-to-many
matches. The result contains 10,043 rows representing 9,957 unique transcript
IDs. Repeated transcript IDs can have different coverage values and labels.

The final positive-label counts are 3,314 for chromatin, 3,314 for cytoplasm,
and 3,312 for polysome. The reconstructed dataframe matches every ordered
value in the
[`morrislab/rna-lifecycle-ietswaart`](https://huggingface.co/datasets/morrislab/rna-lifecycle-ietswaart)
dataset.

### Stored source tables

`ietswaart_wf_transcript_tables.tar.gz` stores the eight transcript tables used
by the dataset builder. Git LFS manages the 4 MB archive, and Python wheels
exclude it. `RNALifecycleIetswaart(force_rebuild_raw=True)` downloads the
archive on demand and runs the table-to-label processing above.

## Gene Ontology source annotations

`go_annotations.tsv.gz` stores the exact selected transcripts and GO
assignments for the molecular-function, biological-process, and
cellular-component tasks. The labels were collected by querying MyGene.info
with human APPRIS gene symbols, retaining all evidence codes, requiring a
direct three-edge `is_a` path from each branch root, and selecting the 20 most
frequent terms.

The MyGene.info records are from spring 2023, while the final class set is
consistent with `go-basic.obo` release 2025-03-16. Historical API responses
were not versioned, so the compact table freezes the published assignments
without retaining the original 1.17 GB padded arrays.

`build_go_dataset()` recreates sequence, CDS, splice, and chromosome features
from GenomeKit's GENCODE v41 annotation. The resulting dataframes match every
ordered value in the published GO parquets.
