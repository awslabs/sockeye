
# exit when any command fails
set -e

DATA_FOLDER=wmt23_data_task_data
# if no dir or symlinked dir exists -- mkdir one
[[ -d $DATA_FOLDER || -L $DATA_FOLDER ]] || mkdir -p ${DATA_FOLDER}
mkdir -p ${DATA_FOLDER}/testsets_v2
for data_set in dev test
do
    for test_set in EMEA EUBookshop Europarl JRC-Acquis
    do
        wget -O ${DATA_FOLDER}/testsets_v2/${test_set}.${data_set}.et-lt.lt https://mtdataexternalpublic.blob.core.windows.net/2023datatask/${data_set}/${test_set}.${data_set}.et-lt.lt
        wget -O ${DATA_FOLDER}/testsets_v2/${test_set}.${data_set}.et-lt.et https://mtdataexternalpublic.blob.core.windows.net/2023datatask/${data_set}/${test_set}.${data_set}.et-lt.et
    done
done

mkdir -p ${DATA_FOLDER}/cosine_similarity
PREFIX="https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-15/cosine_similarity"
for i in {0..9} {a..f}; do
    wget -O "${DATA_FOLDER}/cosine_similarity/cosine_similarity.et-lt.part_$i.tsv.gz" "$PREFIX/cosine_similarity.et-lt.part_$i.tsv.gz"
    wget -O "${DATA_FOLDER}/cosine_similarity/cosine_similarity.lt-et.part_$i.tsv.gz" "$PREFIX/cosine_similarity.lt-et.part_$i.tsv.gz"
done

wget -O ${DATA_FOLDER}/exclude_sent_ids_et-lt.txt https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-15/exclude_sent_ids_et-lt.txt 

mkdir -p ${DATA_FOLDER}/sentences
wget -O ${DATA_FOLDER}/sentences/sentences.et.tsv.gz https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-15/sentences.et.tsv.gz 
wget -O ${DATA_FOLDER}/sentences/sentences.lt.tsv.gz https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-15/sentences.lt.tsv.gz 
