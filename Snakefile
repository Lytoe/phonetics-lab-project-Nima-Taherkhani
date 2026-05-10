configfile: "config/config.yaml"

rule all:
    input:
        "data/interim/phoneme_tokens.csv",
        "data/processed/features_acoustic_norm.csv",
        "results/tables/descriptive_acoustic.csv",
        "results/figures/vowel_chart.png"

rule parse_corpus:
    input:
        raw_dir=directory(config["paths"]["raw_dir"])
    output:
        "data/interim/phoneme_tokens.csv"
    shell:
        "python -m src.phonetics_lab.parse_corpus --raw-dir {input.raw_dir} --out {output}"

rule extract_acoustics:
    input:
        tokens="data/interim/phoneme_tokens.csv"
    output:
        "data/processed/features_acoustic.csv"
    params:
        raw_dir=config["paths"]["raw_dir"],
        n_formants=config["acoustics"]["n_formants"],
        max_formant_female=config["acoustics"]["max_formant_female"],
        max_formant_male=config["acoustics"]["max_formant_male"]
    shell:
        "python -m src.phonetics_lab.extract_acoustics --tokens {input.tokens} --raw-dir {params.raw_dir} --out {output} --n-formants {params.n_formants} --max-formant-female {params.max_formant_female} --max-formant-male {params.max_formant_male}"

rule normalise:
    input:
        "data/processed/features_acoustic.csv"
    output:
        "data/processed/features_acoustic_norm.csv"
    shell:
        "python -m src.phonetics_lab.normalise --acoustic {input} --out {output}"

rule analyse:
    input:
        "data/processed/features_acoustic_norm.csv"
    output:
        table="results/tables/descriptive_acoustic.csv",
        fig="results/figures/vowel_chart.png"
    shell:
        "python -m src.phonetics_lab.analyse_descriptive --features {input} --table {output.table} --fig {output.fig}"
