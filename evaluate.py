import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics


class ConllEvaluator(object):
    START_TAGS = {
        ("B", "B"),
        ("I", "B"),
        ("O", "B"),
        ("O", "I"),
        ("E", "E"),
        ("E", "I"),
        ("O", "E"),
        ("O", "I")
    }
    END_TAGS = {
        ("B", "B"),
        ("B", "O"),
        ("I", "B"),
        ("I", "O"),
        ("E", "E"),
        ("E", "I"),
        ("E", "O"),
        ("I", "O")
    }

    def is_chunk_start(self, ptag, tag, ptag_type, tag_type):
        return (ptag, tag) in self.START_TAGS or \
               (tag != "O" and tag != "." and ptag_type != tag_type)

    def is_chunk_end(self, ptag, tag, ptag_type, tag_type):
        return (ptag, tag) in self.END_TAGS or \
               ptag != 'O' and ptag != '.' and ptag_type != tag_type

    @staticmethod
    def split_tag(tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tag_type = ""
        else:
            tag = s[0]
            tag_type = s[1]
        return tag, tag_type

    def evaluate(self, golds, preds):
        """
        https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
        :param golds:
        :param preds:
        :return:
        """
        correct_chunk = {}
        correct_chunk_cnt = 0
        found_correct = {}
        found_correct_cnt = 0
        found_pred = {}
        found_pred_cnt = 0
        correct_tags = 0
        token_count = 0
        for correct_slot, pred_slot in zip(golds, preds):
            in_correct = False
            last_correct_tag = 'O'
            last_correct_type = ''
            last_pred_tag = 'O'
            last_pred_type = ''
            for c, p in zip(correct_slot, pred_slot):
                correct_tag, correct_type = self.split_tag(c)
                pred_tag, pred_type = self.split_tag(p)

                if in_correct:
                    if self.is_chunk_end(last_correct_tag, correct_tag,
                                         last_correct_type, correct_type) and \
                            self.is_chunk_end(last_pred_tag, pred_tag,
                                              last_pred_type, pred_type) and \
                            last_correct_type == last_pred_type:
                        in_correct = False
                        correct_chunk_cnt += 1
                        if last_correct_type in correct_chunk:
                            correct_chunk[last_correct_type] += 1
                        else:
                            correct_chunk[last_correct_type] = 1
                    elif self.is_chunk_end(last_correct_tag, correct_tag,
                                           last_correct_type, correct_type) != \
                            self.is_chunk_end(last_pred_tag, pred_tag,
                                              last_pred_type, pred_type) or \
                            correct_type != pred_type:
                        in_correct = False

                if self.is_chunk_start(last_correct_tag, correct_tag,
                                       last_correct_type, correct_type) and \
                        self.is_chunk_start(last_pred_tag, pred_tag,
                                            last_pred_type, pred_type) and \
                        correct_type == pred_type:
                    in_correct = True

                if self.is_chunk_start(last_correct_tag, correct_tag,
                                       last_correct_type, correct_type):
                    found_correct_cnt += 1
                    if correct_type in found_correct:
                        found_correct[correct_type] += 1
                    else:
                        found_correct[correct_type] = 1

                if self.is_chunk_start(last_pred_tag, pred_tag,
                                       last_pred_type, pred_type):
                    found_pred_cnt += 1
                    if pred_type in found_pred:
                        found_pred[pred_type] += 1
                    else:
                        found_pred[pred_type] = 1

                if correct_tag == pred_tag and correct_type == pred_type:
                    correct_tags += 1

                token_count += 1
                last_correct_tag = correct_tag
                last_correct_type = correct_type
                last_pred_tag = pred_tag
                last_pred_type = pred_type

            if in_correct:
                correct_chunk_cnt += 1
                if last_correct_type in correct_chunk:
                    correct_chunk[last_correct_type] += 1
                else:
                    correct_chunk[last_correct_type] = 1

        if found_pred_cnt > 0:
            precision = 100 * correct_chunk_cnt / found_pred_cnt
        else:
            precision = 0

        if found_correct_cnt > 0:
            recall = 100 * correct_chunk_cnt / found_correct_cnt
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0

        return {
            "overall": {
                "f1": f1,
                "prec": precision,
                "rec": recall
            }
        }


def replace(items, x, y, z):
    return [y if item == x else z for item in items]


def cast_float(d):
    if isinstance(d, dict):
        return {k: cast_float(v) for k, v in d.items()}
    else:
        return float(d)


def evaluate_intents(golds, preds):
    vocab = set(golds) | set(preds)
    measures = (
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score
    )
    ret = {
        "overall": {
            "acc": metrics.accuracy_score(golds, preds),
            "prec": metrics.precision_score(golds, preds, average="micro"),
            "rec": metrics.recall_score(golds, preds, average="micro"),
            "f1": metrics.f1_score(golds, preds, average="micro"),
        },
        "intents": {
            intent: {
                name: measure(
                    y_true=replace(golds, intent, 1, 0),
                    y_pred=replace(preds, intent, 1, 0),
                    average="binary"
                ) for name, measure in zip(("prec", "rec", "f1"), measures)
            } for intent in vocab
        }
    }
    return cast_float(ret)


def strip_bi(sents):
    def _strip_bi(w):
        if w == "O":
            return w
        tokens = w.split("-")
        if len(tokens) == 1:
            return w
        return tokens[1]
    return [" ".join(_strip_bi(w) for w in sent.split()) for sent in sents]


def evaluate(gold_labels, gold_intents, pred_labels, pred_intents):
    return {
        "intent-classification": evaluate_intents(gold_intents, pred_intents),
        "slot-labeling": ConllEvaluator().evaluate(
            golds=[x.split() for x in gold_labels],
            preds=[x.split() for x in pred_labels]
        ),
        "sentence-understanding": float(metrics.accuracy_score(
            y_true=list(map("/".join, zip(strip_bi(gold_labels), gold_intents))),
            y_pred=list(map("/".join, zip(strip_bi(pred_labels), pred_intents)))
        ))
    }