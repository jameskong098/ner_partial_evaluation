# When an @ token appears labeled with B-X where the following token is B-X, they should be counted
# as one entity, not two. This is important for accurate evaluation, and is corrected with the following script

# apply the script to files from the broad_twitter_corpus before training
DELIM = "-"

def correct_file(path: str, new_path: str) -> str:
    """
    Corrects CoNLL data for the Broad Twitter Corpus. When an @ token appears with a B tag
    and the following token is labeled with a B tag of the same type, make the following
    label I. The @ is considered part of the mention in those cases so it should be labeled as such.
    This is valuable because it affects evaluation metrics.
    """
    with open(path, encoding="utf8") as input:
        with open(new_path, mode = "w", encoding="utf8") as output:
            flag = False
            prev_tag = None
            for line in input:
                if line.strip() == "":
                    # write newline
                    output.write("\n")
                    flag = False
                    prev_tag = None
                else:
                    token, tag = line.rstrip("\n").split("\t")
                    if flag and tag == prev_tag:
                        # this is the case where current tag is changed to I
                        b, entity_type = tag.split(DELIM)
                        tag = DELIM.join(["I", entity_type])
                        flag = False
                        prev_tag = None
                    elif token == "@" and tag.split(DELIM)[0] == "B":
                        flag = True
                        prev_tag = tag
                    output.write("\t".join([token, tag]))
                    output.write("\n")
    
    return new_path