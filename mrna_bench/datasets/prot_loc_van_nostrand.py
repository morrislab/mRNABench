import numpy as np
import pandas as pd
from tqdm import tqdm
import genome_kit as gk

from mrna_bench.datasets.benchmark_dataset import BenchmarkDataset
from mrna_bench.datasets.dataset_utils import ohe_to_str
from mrna_bench.utils import download_file


PL_URL = (
    "https://zenodo.org/records/14708163/files/"
    "protein_localization_dataset.npz"
)


class ProteinLocalizationVan(BenchmarkDataset):
    """Protein Subcellular Localization Dataset."""

    def __init__(self, 
        force_redownload: bool = False,
        **kwargs # noqa
    ):
        """Initialize ProteinLocalization dataset.

        Args:
            force_redownload: Force raw data download even if pre-existing.
        """
        super().__init__(
            dataset_name="prot-loc-van",
            species=["human"],
            force_redownload=force_redownload
        )

    def get_raw_data(self):
        """Download raw data from source."""
        print("Downloading raw data...")
        pass

    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
    def process_raw_data(self) -> pd.DataFrame:
        """Process raw data into Pandas dataframe.

        Returns:
            Pandas dataframe of processed sequences.
        """
        df = pd.read_csv('/home/fradkinp/Documents/01_projects/data_storage/VanNostrand_2020.csv', header=1)
        df1 = df[['Unnamed: 0', 'Unnamed: 1', 'Cytoplasm', 'Nuclei']]
        # 
        df1= df1.loc[~df1.isnull().any(axis=1)].reset_index(drop=True)
        df1 = df1.rename(columns={'Unnamed: 0': 'gene_name', 'Unnamed: 1': 'ensg_id'})
        df1
        six_t_e = []
        labels_cytoplasm = []
        labels_nuclei = []
        genome = gk.Genome('gencode.v29')

        data=[]
        for index, row in tqdm(df1.iterrows(), total=len(df1)):
            gene_name = row['ensg_id']
            genes = [gene for gene in genome.genes if gene.id.split('.')[0] == gene_name]

            gene_name
            if len(genes) == 0:
                print("Can't find gene name", gene_name)
                continue
            if len(genes) == 2:
                print(gene_name)
                print(genes)
                genes = genes[0]

            gk_gene = genes[0]

            transcript = genome.appris_transcripts(gk_gene)[0]
            
            six_t_e.append(create_six_track_encoding(transcript, genome))
            labels_cytoplasm.append(row['Cytoplasm'])
            labels_nuclei.append(row['Nuclei'])
            sequence = "".join([genome.dna(x) for x in transcript.exons])

            target = np.array([int(row['Nuclei']), int(row['Cytoplasm'])])
            data.append(
                {
                    'labels_cytoplasm': int(row['Cytoplasm']),
                    'labels_nuclei': int(row['Nuclei']),
                    'sequence': sequence,
                    'splice': create_splice_track(transcript),
                    'cds': create_cds_track(transcript),
                    'target': target,
                    'gene': transcript.gene.name,
                    'transcript_length': len(sequence)
                }
            )

        df2 = pd.DataFrame(data)

        return df2[['gene','target', 'cds', 'splice', 'sequence', 'transcript_length']]


def find_transcript(genome, transcript_id):
    """Find a transcript in a genome by transcript ID.
    
    Args:
        genome (object): The genome object containing a list of transcripts.
        transcript_id (str): The ID of the transcript to find.
        
    Returns:
        object: The transcript object, if found.
        
    Raises:
        ValueError: If no transcript with the given ID is found.
    
    Example:
        >>> # Create sample transcripts and a genome
        >>> transcript1 = 'ENST00000263946'
        >>> genome = Genome("gencode.v29")
        >>> result = find_transcript(genome, 'ENST00000335137')
        >>> print(result.id)
        <Transcript ENST00000263946.7 of PKP1>
        >>> # If transcript ID is not found
        >>> find_transcript(genome, 'ENST00000000000')
        ValueError: Transcript with ID ENST00000000000 not found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split('.')[0] == transcript_id]
    if not transcripts:
        raise ValueError(f"Transcript with ID {transcript_id} not found.")
    
    return transcripts[0]

def find_transcript_by_gene_name(genome, gene_name):
    """Find all transcripts in a genome by gene name.
    
    Args:
        genome (object): The genome object containing a list of transcripts.
        gene_name (str): The name of the gene whose transcripts are to be found.
        
    Returns:
        list: A list of transcript objects corresponding to the given gene name.
        
    Raises:
        ValueError: If no transcripts for the given gene name are found.
    
    Example:
        >>> # Find transcripts by gene name
        >>> transcripts = find_transcript_by_gene_name(genome, 'PKP1')
        >>> print(transcripts)
        [<Transcript ENST00000367324.7 of PKP1>,
        <Transcript ENST00000263946.7 of PKP1>,
        <Transcript ENST00000352845.3 of PKP1>,
        <Transcript ENST00000475988.1 of PKP1>,
        <Transcript ENST00000477817.1 of PKP1>]        
        >>> # If gene name is not found
        >>> find_transcript_by_gene_name(genome, 'XYZ')
        ValueError: No transcripts found for gene name XYZ.
    """
    genes = [x for x in genome.genes if x.name == gene_name]
    if not genes:
        raise ValueError(f"No genes found for gene name {gene_name}.")
    if len(genes) > 1:
        print(f"Warning: More than one gene found for gene name {gene_name}.")
        print('Concatenating transcripts from all genes.')
        
    transcripts = []
    for gene in genes:
        transcripts += gene.transcripts
    return transcripts

def create_cds_track(t):
    """Create a track of the coding sequence of a transcript.
    Use the exons of the transcript to create a track where the first position of the codon is one.
    
    Args:
        t (gk.Transcript): The transcript object.
    """
    cds_intervals = t.cdss
    utr3_intervals = t.utr3s
    utr5_intervals = t.utr5s
    
    len_utr3 = sum([len(x) for x in utr3_intervals])
    len_utr5 = sum([len(x) for x in utr5_intervals])
    len_cds = sum([len(x) for x in cds_intervals])
    
    # create a track where first position of the codon is one
    cds_track = np.zeros(len_cds, dtype=int)
    # set every third position to 1
    cds_track[0::3] = 1
    # concat with zeros of utr3 and utr5
    cds_track = np.concatenate([np.zeros(len_utr5, dtype=int), cds_track, np.zeros(len_utr3, dtype=int)])
    return cds_track

def create_splice_track(t):
    """Create a track of the splice sites of a transcript.
    The track is a 1D array where the positions of the splice sites are 1.

    Args:
        t (gk.Transcript): The transcript object.
    """
    len_utr3 = sum([len(x) for x in t.utr3s])
    len_utr5 = sum([len(x) for x in t.utr5s])
    len_cds = sum([len(x) for x in t.cdss])
    
    len_mrna = len_utr3 + len_utr5 + len_cds
    splicing_track = np.zeros(len_mrna, dtype=int)
    cumulative_len = 0
    for exon in t.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1:cumulative_len] = 1
        
    return splicing_track

# convert to one hot
def seq_to_oh(seq):
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        if base == 'A':
            oh[i, 0] = 1
        elif base == 'C':
            oh[i, 1] = 1
        elif base == 'G':
            oh[i, 2] = 1
        elif base == 'T':
            oh[i, 3] = 1
    return oh

def create_one_hot_encoding(t, genome):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.

    Args
        t (gk.Transcript): The transcript object.
    """
    seq = "".join([genome.dna(exon) for exon in t.exons])
    oh = seq_to_oh(seq)
    return oh

def create_six_track_encoding(t, genome, channels_last=False):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.
    Concatenate the one-hot encoding with the cds track and the splice track.

    Args
        t (gk.Transcript): The transcript object.
    """
    oh = create_one_hot_encoding(t, genome)
    cds_track = create_cds_track(t)
    splice_track = create_splice_track(t)
    six_track = np.concatenate([oh, cds_track[:, None], splice_track[:, None]], axis=1)
    if not channels_last:
        six_track = six_track.T
    return six_track