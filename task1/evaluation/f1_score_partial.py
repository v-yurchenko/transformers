from typing import List, Optional, Type, Set, Tuple, Union, Callable
import enum
from collections import defaultdict
import numpy as np
import warnings
from itertools import chain
from sklearn.exceptions import UndefinedMetricWarning

# The code is a modified version of https://github.com/chakki-works/seqeval


class Tag(enum.Flag):
    SAME = enum.auto()
    DIFF = enum.auto()
    ANY = SAME | DIFF


class Entity:

    def __init__(self, sent_id: int, start: int, end: int, tag: str):
        self.sent_id = sent_id
        self.start = start
        self.end = end
        self.tag = tag

    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.sent_id, self.tag, self.start, self.end)

    def __eq__(self, other: 'Entity'):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        return self.sent_id, self.tag, self.start, self.end


class Prefix(enum.Flag):
    I = enum.auto()
    O = enum.auto()
    B = enum.auto()
    E = enum.auto()
    S = enum.auto()
    U = enum.auto()
    L = enum.auto()
    ANY = I | O | B | E | S | U | L


Prefixes = dict(Prefix.__members__)


class Token:
    allowed_prefix = None
    start_patterns = None
    inside_patterns = None
    end_patterns = None

    def __init__(self, token: str, suffix: bool = False, delimiter: str = '-'):
        self.token = token
        self.prefix = Prefixes[token[-1]] if suffix else Prefixes[token[0]]
        tag = token[:-1] if suffix else token[1:]
        self.tag = tag.strip(delimiter) or '_'

    def __repr__(self):
        return self.token

    def is_valid(self):
        """Check whether the prefix is allowed or not."""
        if self.prefix not in self.allowed_prefix:
            allowed_prefixes = str(self.allowed_prefix).replace('Prefix.', '')
            message = 'Invalid token is found: {}. Allowed prefixes are: {}.'
            raise ValueError(message.format(self.token, allowed_prefixes))
        return True

    def is_start(self, prev: 'Token'):
        """Check whether the current token is the start of chunk."""
        return self.check_patterns(prev, self.start_patterns)

    def is_inside(self, prev: 'Token'):
        """Check whether the current token is inside of chunk."""
        return self.check_patterns(prev, self.inside_patterns)

    def is_end(self, prev: 'Token'):
        """Check whether the previous token is the end of chunk."""
        return self.check_patterns(prev, self.end_patterns)

    def check_tag(self, prev: 'Token', cond: Tag):
        """Check whether the tag pattern is matched."""
        if cond == Tag.ANY:
            return True
        if prev.tag == self.tag and cond == Tag.SAME:
            return True
        if prev.tag != self.tag and cond == Tag.DIFF:
            return True
        return False

    def check_patterns(self, prev: 'Token', patterns: Set[Tuple[Prefix, Prefix, Tag]]):
        """Check whether the prefix patterns are matched."""
        for prev_prefix, current_prefix, tag_cond in patterns:
            if prev.prefix in prev_prefix and self.prefix in current_prefix and self.check_tag(prev, tag_cond):
                return True
        return False


class Tokens:

    def __init__(self, tokens: List[str], scheme: Type[Token],
                 suffix: bool = False, delimiter: str = '-', sent_id: int = None):
        self.outside_token = scheme('O', suffix=suffix, delimiter=delimiter)
        self.tokens = [scheme(token, suffix=suffix, delimiter=delimiter) for token in tokens]
        self.extended_tokens = self.tokens + [self.outside_token]
        self.sent_id = sent_id

    @property
    def entities(self):
        """Extract entities from tokens.
        Returns:
            list: list of Entity.
        Example:
            >>> tokens = Tokens(['B-PER', 'I-PER', 'O', 'B-LOC'], IOB2)
            >>> tokens.entities
            [('PER', 0, 2), ('LOC', 3, 4)]
        """
        i = 0
        entities = []
        prev = self.outside_token
        while i < len(self.extended_tokens):
            token = self.extended_tokens[i]
            token.is_valid()
            if token.is_start(prev):
                end = self._forward(start=i + 1, prev=token)
                if self._is_end(end):
                    entity = Entity(sent_id=self.sent_id, start=i, end=end, tag=token.tag)
                    entities.append(entity)
                i = end
            else:
                i += 1
            prev = self.extended_tokens[i - 1]
        return entities

    def _forward(self, start: int, prev: Token):
        for i, token in enumerate(self.extended_tokens[start:], start):
            if token.is_inside(prev):
                prev = token
            else:
                return i
        return len(self.tokens) - 1

    def _is_end(self, i: int):
        token = self.extended_tokens[i]
        prev = self.extended_tokens[i - 1]
        return token.is_end(prev)


class Entities:

    def __init__(self, sequences: List[List[str]], scheme: Type[Token], suffix: bool = False, delimiter: str = '-'):
        self.entities = [
            Tokens(seq, scheme=scheme, suffix=suffix, delimiter=delimiter, sent_id=sent_id).entities
            for sent_id, seq in enumerate(sequences)
        ]

    def filter(self, tag_name: str):
        entities = {entity for entity in chain(*self.entities) if entity.tag == tag_name}
        return entities

    @property
    def unique_tags(self):
        tags = {
            entity.tag for entity in chain(*self.entities)
        }
        return tags


class IOB1(Token):
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    start_patterns = {
        (Prefix.O, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.I, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.SAME),
        (Prefix.B, Prefix.B, Tag.SAME)
    }
    inside_patterns = {
        (Prefix.B, Prefix.I, Tag.SAME),
        (Prefix.I, Prefix.I, Tag.SAME)
    }
    end_patterns = {
        (Prefix.I, Prefix.I, Tag.DIFF),
        (Prefix.I, Prefix.O, Tag.ANY),
        (Prefix.I, Prefix.B, Tag.ANY),
        (Prefix.B, Prefix.O, Tag.ANY),
        (Prefix.B, Prefix.I, Tag.DIFF),
        (Prefix.B, Prefix.B, Tag.SAME)
    }


PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = 'sample', 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0
    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s. Use `zero_division` parameter to control'
           ' this behavior.'.format(msg_start, modifier, axis0))
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


def unique_labels(y_true: List[List[str]], y_pred: List[List[str]],
                  scheme: Type[Token], suffix: bool = False) -> List[str]:
    sequences_true = Entities(y_true, scheme, suffix)
    sequences_pred = Entities(y_pred, scheme, suffix)
    unique_tags = sequences_true.unique_tags | sequences_pred.unique_tags
    return sorted(unique_tags)


def get_tp_sum_old(ent_true, ent_pred):
    intersection = [[0 for i in ent_pred] for j in ent_true]
    for idx_t, e_t in enumerate(ent_true):
        for idx_p, e_p in enumerate(ent_pred):
            l_int = len(set(e_t) & set(e_p))
            intersection[idx_t][idx_p] = l_int/len(set(e_t))
    max_els = []
    for var in intersection:
        max_els.append(np.argmax(var))
    all_tp = sum([intersection[i][max_els[i]] for i in range(len(intersection))])
    return all_tp


def get_tp_sum(ent_true, ent_pred, verbose=False):
    #if verbose:
    #    print('Entity: ', ent_pred)
    intersection = [[0 for i in ent_pred] for j in ent_true]
    for idx_t, e_t in enumerate(ent_true):
        for idx_p, e_p in enumerate(ent_pred):
            cmp_t = np.arange(e_t[0], e_t[1]+1)
            cmp_p = np.arange(e_p[0], e_p[1]+1)
            l_int = len(set(cmp_t) & set(cmp_p))
            intersection[idx_t][idx_p] = l_int/max(len(set(cmp_t)), len(set(cmp_p)))
    max_els = []
    for var in intersection:
        max_els.append(np.argmax(var))
    used = set()
    new_max_els = []
    for m in max_els:
        if m in used:
            new_max_els.append(None)
        else:
            new_max_els.append(m)
            used.add(m)
    all_tp = sum([intersection[i][new_max_els[i]] if new_max_els[i] is not None else 0
                  for i in range(len(intersection))])
    return all_tp


def extract_tp_actual_correct(y_true, y_pred, suffix, partial):
    #print('PARTIAL VALUE: ', partial)
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:

        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        verbose = False
        #if type_name == 'Aspect':
        #    verbose = True
        #    for ent in entities_true_type:
        #        if ent[0] == 1764 or ent[0] == 3006:
        #            print('Matching aspect: ', ent)
        if not partial:
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        else:
            tp_sum = np.append(tp_sum, get_tp_sum(entities_true_type, entities_pred_type, verbose=verbose))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """Check that all arrays have consistent first and second dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if not is_list == {list}:
        raise TypeError('Found input variables without list of list.')

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)


def _precision_recall_fscore_support(y_true: List[List[str]],
                                     y_pred: List[List[str]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     scheme: Optional[Type[Token]] = None,
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None,
                                     partial: bool = False) -> SCORES:
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    check_consistent_length(y_true, y_pred)

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, suffix, partial)
    #print(f'Predicted sum: {pred_sum}, true sum: {true_sum}, true positive sum: {tp_sum}')

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)
    #print('True sum: ', true_sum)
    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False,
                                    partial: bool = False) -> SCORES:
    """Compute precision, recall, F-measure and support for each class.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.
        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.
        suffix : bool, False by default.
    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]
        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]
        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]
        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.
    Examples:
        >>> from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        (0.5, 0.5, 0.5, 2)
        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:
        >>> precision_recall_fscore_support(y_true, y_pred, average=None)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))
    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct,
        partial=partial
    )

    return precision, recall, f_score, true_sum


def f1_score(y_true: List[List[str]], y_pred: List[List[str]],
             *,
             average: Optional[str] = 'micro',
             suffix: bool = False,
             mode: Optional[str] = None,
             sample_weight: Optional[List[int]] = None,
             zero_division: str = 'warn',
             scheme: Optional[Type[Token]] = None,
             partial: bool = False):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        average : string, [None, 'micro' (default), 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.
        mode : str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.
        scheme : Token, [IOB2, IOE2, IOBES]
        suffix : bool, False by default.
    Returns:
        score : float or array of float, shape = [n_unique_labels].
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred, average='micro')
        0.6666666666666666
        >>> f1_score(y_true, y_pred, average='macro')
        0.75
        >>> f1_score(y_true, y_pred, average='weighted')
        0.6666666666666666
        >>> f1_score(y_true, y_pred, average=None)
        array([0.5, 1. ])
    """

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 beta=1,
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division,
                                                 suffix=suffix,
                                                 partial=partial)
    return f, s