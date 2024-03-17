# Helper function 3: download text classification dataset from AllenNLP repo
tmp_dir=data/original
out_dir=data/original

mkdir -p $out_dir/text_cat

function download_allennlp {
    local name=$1
    local remote_name=$2
    local label_delta=$3

    mkdir -p $tmp_dir/$name/
    mkdir -p $out_dir/text_cat/$name/

    for i in train dev test; do
        json_file=$tmp_dir/$name/${i}.jsonl
        curl -Lo $json_file https://s3-us-west-2.amazonaws.com/allennlp/datasets/$remote_name/${i}.jsonl
        cat $json_file | jq -r '[.label, .text] | @tsv' | awk -v label_delta=$label_delta -F $'\t' '{print $1-label_delta"\t"$2}' | sed 's/\\\\/\\/g' >$out_dir/text_cat/$name/$i
    done
}

#### AG news and IMDB
download_allennlp ag ag-news 1
download_allennlp imdb imdb 0
