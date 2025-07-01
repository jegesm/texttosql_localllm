SELECT pg_catalog.set_config('search_path', '', false);
CREATE SCHEMA distilled;
CREATE TYPE distilled.type_assembly AS ENUM (
    'single_sample_assembly',
    'coassembly'
);
CREATE TYPE distilled.type_classification_method AS ENUM (
    'ani_screen',
    'taxonomic classification defined by topology and ANI',
    'taxonomic novelty determined using RED',
    'taxonomic classification fully defined by topology'
);
CREATE TYPE distilled.type_completeness_model AS ENUM (
    'Neural Network (Specific Model)',
    'Gradient Boost (General Model)'
);
CREATE TYPE distilled.type_contig AS ENUM (
    'chromosome',
    'plasmid',
    'phage'
);
CREATE TYPE distilled.type_genetype AS ENUM (
    'CDS',
    'rRNA',
    'tRNA',
    'tmRNA'
);
CREATE TYPE distilled.type_phenotype AS ENUM (
    'Ceftriaxone',
    'Chloramphenicol',
    'Tylosin',
    'Meropenem',
    'Avibactam',
    'Penicillin',
    'Trimethoprim',
    'Nalidixic acid',
    'Cephalotin',
    'Tobramycin',
    'Pristinamycin IIA',
    'Ceftazidime',
    'Oleandomycin',
    'Streptomycin',
    'Aztreonam',
    'Doxycycline',
    'Telithromycin',
    'Virginiamycin M',
    'Cephalothin',
    'Virginiamycin S',
    'Fosfomycin',
    'Tigecycline',
    'Ciprofloxacin',
    'Tiamulin',
    'Imipenem',
    'Clavulanic acid',
    'Pristinamycin IA',
    'Mupirocin',
    'Erythromycin',
    'Dalfopristin',
    'Cefixime',
    'Amoxicillin',
    'Cefotaxime',
    'Amikacin',
    'Piperacillin',
    'Colistin',
    'Tazobactam',
    'Cefepime',
    'Fusidic acid',
    'Minocycline',
    'Lincomycin',
    'Gentamicin',
    'Quinupristin',
    'Carbomycin',
    'Spiramycin',
    'Vancomycin',
    'Tetracycline',
    'Linezolid',
    'Temocillin',
    'Cefoxitin',
    'Sulfamethoxazole',
    'Florfenicol',
    'Azithromycin',
    'Clindamycin',
    'Ertapenem',
    'Ampicillin',
    'Teicoplanin',
    'Ticarcillin'
);
CREATE TYPE distilled.type_rank AS ENUM (
    'genus',
    'species',
    'family',
    'phylum',
    'order',
    'class',
    'kingdom',
    'no rank'
);

CREATE TABLE distilled.antibiotic_resfinder (
    id integer NOT NULL,
    gene_resfinder_id integer,
    antibiotic distilled.type_phenotype
);
CREATE TABLE distilled.checkm2 (
    id integer NOT NULL,
    mag_reference_id integer,
    run integer,
    completeness_model distilled.type_completeness_model,
    completeness_general real,
    completeness_specific real,
    contamination real,
    translation_table_used integer,
    coding_density real,
    contig_n50 integer,
    average_gene_length real,
    genome_size integer,
    gc_content real,
    total_coding_sequences integer,
    additional_notes text
);

CREATE TABLE distilled.class_resfinder (
    id integer NOT NULL,
    name text,
    gene_resfinder_ids integer[]
);

CREATE TABLE distilled.class_resfinder_clr (
    meta_id integer,
    run integer,
    class_resfinder_id integer,
    clr integer
);

CREATE TABLE distilled.gene_resfinder_clr (
    meta_id integer,
    run integer,
    gene_resfinder_id integer,
    clr integer

); /* gives quantities for the abundance of  genes in samples through the meta table */

CREATE TABLE distilled.gene_resfinder (
    id integer NOT NULL,
    name text,
    length integer,
    enzymatic_inactivation boolean,
    enzymatic_modification boolean,
    increased_efflux boolean,
    target_modification boolean,
    target_protection boolean
);

CREATE TABLE distilled.contig (
    id integer NOT NULL,
    location_id integer,
    mag_reference_id integer,
    name character varying(16),
    length integer,
    coverage integer,
    possible_source detailed.type_contig
);
CREATE TABLE distilled.gene_annotation (
    id integer NOT NULL,
    contig_id integer,
    gene distilled.type_gene,
    protein_id integer,
    genetype distilled.type_genetype,
    ec_number character varying(16),
    cog_reference character varying(8),
    pos_start integer,
    pos_stop integer,
    length integer,
    reverse boolean
);

CREATE TABLE distilled.gtdb (
    id integer NOT NULL,
    mag_reference_id integer,
    tax_gtdb_id integer,
    classification_method distilled.type_classification_method,
    msa_percent real,
    translation_table real,
    red_value real
);
CREATE TABLE distilled.location (
    id integer NOT NULL,
    country character varying(16),
    city character varying(32),
    plant character varying(48),
    lat double precision,
    lon double precision
);
CREATE TABLE distilled.mag_abundance_clr (
    mag_reference_id integer,
    meta_id integer,
    clr double precision
);

CREATE TABLE distilled.mag_reference (
    id integer NOT NULL,
    bin_number integer,
    run integer,
    name character varying(64),
    assembly_type distilled.type_assembly
);
CREATE TABLE distilled.meta (
    id integer NOT NULL,
    sample_id integer,
    location_id integer,
    collection_date date,
    complete_name character varying(64),
    lockdown boolean,
    ph double precision,
    temperature double precision
);
CREATE TABLE distilled.protein (
    id integer NOT NULL,
    description character varying(128)
);
CREATE TABLE distilled.qpcr (
    meta_id integer,
    qpcr_target_id integer,
    tax_gtdb_id integer,
    pathogen_name character varying(96),
    is_present boolean,
    ct double precision
);

CREATE TABLE distilled.relationship_gtdb (
    tax_gtdb_id integer NOT NULL,
    rank distilled.type_rank,
    tax_gtdb_id_parent integer
);
CREATE TABLE distilled.tax_gtdb (
    id integer NOT NULL,
    name character varying(40),
    rank distilled.type_rank,
    human_microbiome boolean,
    ncbi_reference character varying(16)
);

CREATE TABLE distilled.qpcr_target (
    id integer NOT NULL,
    target varchar(8),
    forward_primer varchar(16),
    reverse_primer varchar(16),
    probe varchar(16),
    forward_primer_sequence text,
    reverse_primer_sequence text,
    probe_sequence text
);

CREATE TABLE distilled.mag_presence_absence (
    meta_id integer,
    location_id integer,
    tax_gtdb_id integer,
    is_present boolean
);

