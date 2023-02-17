import random


def get_mscxr_synth_dataset(opt, dataset, label_key="finding_labels", finding_key="label_text"):
    n = opt.n_synth_samples_per_class

    synth_dataset = {}

    for i in range(len(dataset)):

        sample = dataset[i]
        label = sample[label_key]
        if str(label) == "nan":
            continue

        label = random.choice(label.split("|"))
        if synth_dataset.get(label) is None:
            synth_dataset[label] = []

        label_texts = sample[finding_key]
        for label_text in label_texts.split("|"):
            synth_dataset[label].append(label_text)

    for label in synth_dataset.keys():
        random.shuffle(synth_dataset[label])
        while len(synth_dataset[label]) < n:
            synth_dataset[label] = synth_dataset[label] + synth_dataset[label]
        synth_dataset[label] = synth_dataset[label][:n]

    new_dataset = []
    for label in synth_dataset.keys():
        for label_text in synth_dataset[label]:
            new_dataset.append({label: label_text})
    return new_dataset, list(synth_dataset.keys())