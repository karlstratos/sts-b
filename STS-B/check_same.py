from random import randint

def get_new_examples(fname):
    examples = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:  # Attribute line
                continue
            pieces = line.split('\t')
            sent1 = pieces[7].strip()
            sent2 = pieces[8].strip()
            score = float(pieces[9]) if len(pieces) > 9 else None
            examples.append((sent1, sent2, score))
    return examples

def get_old_examples(fname):
    examples = []
    with open(fname) as f:
        for line in f:
            pieces = line.split('\t')
            sent1 = pieces[5].strip()
            sent2 = pieces[6].strip()
            score = float(pieces[4])
            examples.append((sent1, sent2, score))
    return examples

# Check train
new_examples = get_new_examples('train.tsv')
old_examples = get_old_examples('original/sts-train.tsv')
nassert len(new_examples) == len(old_examples)
for i in range(len(new_examples)):
    assert new_examples[i] == old_examples[i]
print('[TRAIN] passed: %d examples exactly the same' % len(new_examples))
sent1, sent2, score = new_examples[randint(0, len(new_examples) - 1)]
print('  sent1: %s' % sent1)
print('  sent2: %s' % sent2)
print('  score: %g' % score)

# Check dev
print()
new_examples = get_new_examples('dev.tsv')
old_examples = get_old_examples('original/sts-dev.tsv')
assert len(new_examples) == len(old_examples)
for i in range(len(new_examples)):
    assert new_examples[i] == old_examples[i]
print('[DEV] passed: %d examples exactly the same' % len(new_examples))
sent1, sent2, score = new_examples[randint(0, len(new_examples) - 1)]
print('  sent1: %s' % sent1)
print('  sent2: %s' % sent2)
print('  score: %g' % score)

# Check test
print()
new_examples = get_new_examples('test.tsv')
old_examples = get_old_examples('original/sts-test.tsv')
assert len(new_examples) == len(old_examples)
for i in range(len(new_examples)):  # Test portion does not have scores in new
    assert new_examples[i][0] == old_examples[i][0]
    assert new_examples[i][1] == old_examples[i][1]
print('[TEST] passed: %d examples exactly the same' % len(new_examples))
sent1, sent2, score = old_examples[randint(0, len(new_examples) - 1)]
print('  sent1: %s' % sent1)
print('  sent2: %s' % sent2)
print('  score: %g' % score)
