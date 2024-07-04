import os
import sys
import csv


def main():
    input_path = sys.argv[1]
    if input_path[-1] != "/":
        input_path += "/"
        
    lang_1 = input_path.split("_")[0]
    lang_2 = input_path.split("_")[1].replace("/", "")
    (
        ent_ill,
        train_ill,
        test_ill,
        index2rel,
        index2entity,
        rel2index,
        entity2index,
        rel_triples_1,
        rel_triples_2,
        entname_1,
        entname_2,
    ) = read_data(input_path)

    rel_triples_1 = replace_entity_id(rel_triples_1, index2rel, index2entity)
    rel_triples_2 = replace_entity_id(rel_triples_2, index2rel, index2entity)

    ent_ill = replace_entity_id_in_pair(ent_ill, index2entity)

    output_path = f"{input_path}converted/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + "rel_triples_1", "w") as f:
        for h, r, t in rel_triples_1:
            f.write(f"{h}\t{r}\t{t}\n")

    with open(output_path + "rel_triples_2", "w") as f:
        for h, r, t in rel_triples_2:
            f.write(f"{h}\t{r}\t{t}\n")

    with open(output_path + "ent_links", "w") as f:
        for h, t in ent_ill:
            f.write(f"{h}\t{t}\n")
            
    with open(output_path + "ent_ids_1", "w") as f:
        for n in entname_1:
            f.write(f"{n}\n")
    
    with open(output_path + "ent_ids_2", "w") as f:
        for n in entname_2:
            f.write(f"{n}\n")

    replace_space_with_tab(
        input_path + f"{lang_1}_att_triples", output_path + "attr_triples_1"
    )
    replace_space_with_tab(
        input_path + f"{lang_2}_att_triples", output_path + "attr_triples_2"
    )


def replace_space_with_tab(file_path, output_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=" ", escapechar="\\")
        with open(output_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in reader:
                row = [*map(lambda line: line.strip().strip("<>"), row[:-1]), row[-1]]
                writer.writerow(
                    filter(lambda x: x.strip() != "." and x.strip() != "", row)
                )


def replace_entity_id(triples, index2rel, index2entity):
    new_triples = []
    for h, r, t in triples:
        new_triples.append((index2entity[h], index2rel[r], index2entity[t]))
    return new_triples


def replace_entity_id_in_pair(pairs, index2entity):
    new_pairs = []
    for h, t in pairs:
        new_pairs.append((index2entity[h], index2entity[t]))
    return new_pairs


def read_data(data_path: str):
    def read_idtuple_file(file_path):
        print("loading a idtuple file...   " + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip("\n").split("\t")
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret

    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                print("loading a (id2object)file...  " + file_path)
                for line in f:
                    th = line.strip("\n").split("\t")
                    id2object[int(th[0])] = th[1]
        return id2object

    def read_idobj_tuple_file(file_path):
        print("loading a idx_obj file...   " + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip("\n").split("\t")
                ret.append((int(th[0]), th[1]))
        return ret

    print("load data from... :", data_path)
    # ent_index(ent_id)2entity / relation_index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1", data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1", data_path + "rel_ids_2"])
    entity2index = {e: idx for idx, e in index2entity.items()}
    rel2index = {r: idx for idx, r in index2rel.items()}

    # triples
    rel_triples_1 = read_idtuple_file(data_path + "triples_1")
    rel_triples_2 = read_idtuple_file(data_path + "triples_2")
    index_with_entity_1 = read_idobj_tuple_file(data_path + "ent_ids_1")
    index_with_entity_2 = read_idobj_tuple_file(data_path + "ent_ids_2")

    # ill
    train_ill = read_idtuple_file(data_path + "sup_pairs")
    test_ill = read_idtuple_file(data_path + "ref_pairs")
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)

    # ent_idx
    entid_1 = [entid for entid, _ in index_with_entity_1]
    entid_2 = [entid for entid, _ in index_with_entity_2]
    entids = list(range(len(index2entity)))

    entname_1 = [name for _, name in index_with_entity_1]
    entname_2 = [name for _, name in index_with_entity_2]

    return (
        ent_ill,
        train_ill,
        test_ill,
        index2rel,
        index2entity,
        rel2index,
        entity2index,
        rel_triples_1,
        rel_triples_2,
        entname_1,
        entname_2,
    )


if __name__ == "__main__":
    main()
