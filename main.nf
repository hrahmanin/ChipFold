nextflow.enable.dsl=2

params.sra_id   = params.sra_id   ?: 'SRR058989'
params.outdir   = params.outdir   ?: 'results'
params.ref_url  = params.ref_url  ?: 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz'

process FETCH_FASTQ {
  tag "$sra_id"
  publishDir params.outdir, mode: 'copy', pattern: '*.fastq.gz'

  input:
  val sra_id

  output:
  tuple val(sra_id), path("${sra_id}_*.fastq.gz")

  shell:
  '''
  set -euo pipefail
  acc='!{sra_id}'
  prefix="${acc:0:6}"
  last3="${acc: -3}"

  baseA="https://ftp.sra.ebi.ac.uk/vol1/fastq/${prefix}/${last3}/${acc}"
  baseB="https://ftp.sra.ebi.ac.uk/vol1/fastq/${prefix}/${acc}"

  pick=""
  for b in "$baseA" "$baseB"; do
    if curl -sfI "$b/${acc}_1.fastq.gz" >/dev/null || curl -sfI "$b/${acc}.fastq.gz" >/dev/null; then
      pick="$b"; break
    fi
  done
  [ -n "$pick" ] || { echo "ENA URL not found for $acc" >&2; exit 1; }

  if curl -sfI "$pick/${acc}_1.fastq.gz" >/dev/null; then
    curl -fL -o "${acc}_1.fastq.gz" "$pick/${acc}_1.fastq.gz"
    if curl -sfI "$pick/${acc}_2.fastq.gz" >/dev/null; then
      curl -fL -o "${acc}_2.fastq.gz" "$pick/${acc}_2.fastq.gz"
    fi
  else
    curl -fL -o "${acc}_1.fastq.gz" "$pick/${acc}.fastq.gz"
  fi
  '''
}

process FASTQC {
  tag "${fastq.simpleName}"
  publishDir "${params.outdir}/fastqc", mode: 'copy', pattern: '*_fastqc.*'

  input:
  path fastq

  output:
  path "${fastq.simpleName}_fastqc.zip"
  path "${fastq.simpleName}_fastqc.html"

  shell:
  '''
  set -euo pipefail
  fastqc -t 1 "!{fastq}"
  '''
}

process ALIGN_BT2 {
  tag "$sample_id"
  publishDir "${params.outdir}/align", mode: 'copy', pattern: '*.sorted.bam*'

  input:
  tuple val(sample_id), path(reads)

  output:
  path "${sample_id}.sorted.bam"
  path "${sample_id}.sorted.bam.bai"

  shell:
  '''
  set -euo pipefail
  acc='!{sample_id}'
  url='!{params.ref_url}'

  # Get reference FASTA inside the task dir to avoid host mounts
  if [ ! -s "mm10.fa" ]; then
    curl -fL -o mm10.fa.gz "$url"
    gunzip -c mm10.fa.gz > mm10.fa
  fi

  # Build local bowtie2 index
  idx="genome_index"
  if [ ! -e "${idx}.1.bt2" ] && [ ! -e "${idx}.1.bt2l" ]; then
    bowtie2-build mm10.fa "$idx"
  fi

  set -- !{reads}
  if [ "$#" -eq 2 ]; then
    bowtie2 -x "$idx" -1 "$1" -2 "$2" -S "${acc}.sam"
  else
    bowtie2 -x "$idx" -U "$1" -S "${acc}.sam"
  fi

  samtools view -S -b "${acc}.sam" | samtools sort -o "${acc}.sorted.bam"
  samtools index "${acc}.sorted.bam"
  '''
}

workflow {
  ch_fetch = FETCH_FASTQ( Channel.value(params.sra_id) )

  ch_fetch.flatMap { sid, files -> files } | FASTQC

  ch_align = ch_fetch.groupTuple().map { sid, files -> tuple(sid, files) }
  ALIGN_BT2( ch_align )
}
