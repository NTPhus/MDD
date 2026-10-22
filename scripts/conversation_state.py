"""Conservative cafe-order facts grounded in the learner's actual turns.

This is a small, explicit dialogue frame, not a semantic summary or a model.
Unknown/mixed-topic messages leave the frame rather than inventing facts.
"""
from __future__ import annotations

import re
import unicodedata


_BEVERAGES = (
    ('hot chocolate', r'hot\s+chocolates?'), ('milk tea', r'milk\s+teas?'),
    ('bubble tea', r'bubble\s+teas?'), ('boba tea', r'boba\s+teas?'),
    ('green tea', r'green\s+teas?'), ('black tea', r'black\s+teas?'),
    ('orange juice', r'orange\s+juices?'), ('apple juice', r'apple\s+juices?'),
    ('coffee', r'coffees?'), ('tea', r'teas?'), ('latte', r'lattes?'),
    ('espresso', r'espressos?'), ('cappuccino', r'cappuccinos?'),
    ('americano', r'americanos?'), ('mocha', r'mochas?'),
    ('water', r'waters?'), ('juice', r'juices?'),
)
_DRINKS = re.compile(r'\b(?:' + '|'.join(f'(?P<item{index}>{pattern})' for index, (_, pattern) in enumerate(_BEVERAGES)) + r')\b')
_ORDER_PREFIX = re.compile(
    r'^(?:(?:actually|okay|ok|well|so|instead)\s*[,!:]?\s+)*'
    r'(?P<phrase>i would like(?: to order)?|i want(?: to order)?|i will have|i will take|can i have|can i get|could i have|may i have|'
    r'make (?:it|that)|change my order to)\b\s*')
_DETAIL_PREFIX = re.compile(r'^(?:i would like|i want|i will have|i will take|make)\b\s*')
_FILLER = {'a', 'an', 'the', 'some', 'one', 'two', 'three', 'four', 'five', 'six', '1', '2', '3', '4', '5', '6',
           'cup', 'cups', 'glass', 'glasses', 'bottle', 'bottles', 'of', 'and', 'with', 'please', 'thanks',
           'thank', 'you', 'both', 'each', 'just', 'only', 'instead', 'for', 'me', 'us', 'it', 'them',
           'that', 'those', 'these', 'drinks', 'drink', 'are', 'is', 'be'}
_TEMPERATURE = {'hot': 'hot', 'warm': 'warm', 'iced': 'iced', 'cold': 'cold'}
_SIZE = {'small': 'small', 'medium': 'medium', 'large': 'large', 'regular': 'regular'}
_SERVICE = ((re.compile(r'\b(?:for here|dine in|eat in)\b'), 'for here'),
            (re.compile(r'\b(?:to go|take away|takeaway|take out|takeout)\b'), 'to go'))
_NEGATIVE_OR_UNCERTAIN = re.compile(
    r'\b(?:or|not|no|never|without|maybe|perhaps|might|would have|if|should|recommend|suggest|'
    r'cancel|forget|stop|dont|don\x27t|cannot|can\x27t|rather)\b')
_CANCEL = re.compile(r'\b(?:cancel (?:my |the )?order|forget (?:my |the )?order|never mind|nevermind|'
                     r'changed? my mind|nothing else|no thanks|no thank you)\b')


def _normal(text):
    text = unicodedata.normalize('NFKC', text).replace('’', "'").casefold()
    for before, after in (("i'd", 'i would'), ("i'll", 'i will'), ("what's", 'what is')):
        text = re.sub(r'\b' + re.escape(before) + r'\b', after, text)
    return re.sub(r'\s+', ' ', text.replace('-', ' ')).strip()


def _qualifiers(text, *, unknown_conflicts=False):
    """Remove only explicit supported details; conflicting values are unknown."""
    found = {}
    for slot, choices in (('temperature', _TEMPERATURE), ('size', _SIZE)):
        values = set()
        def remove(match):
            values.add(choices[match.group()])
            return ' '
        text = re.sub(r'\b(?:' + '|'.join(choices) + r')\b', remove, text)
        if len(values) > 1 and not unknown_conflicts:
            return None, text
        if len(values) == 1:
            found[slot] = next(iter(values))
    values = set()
    for pattern, value in _SERVICE:
        if pattern.search(text):
            values.add(value)
            text = pattern.sub(' ', text)
    if len(values) > 1 and not unknown_conflicts:
        return None, text
    if len(values) == 1:
        found['service'] = next(iter(values))
    return found, text


def _only_filler(text):
    return all(token in _FILLER for token in re.findall(r"[^\W_]+(?:'[^\W_]+)*", text))


def _selection(text, *, allow_short=False):
    normalized = _normal(text)
    prefix = _ORDER_PREFIX.match(normalized)
    explicit_request = prefix and prefix['phrase'] in {'can i have', 'can i get', 'could i have', 'may i have'}
    if _NEGATIVE_OR_UNCERTAIN.search(normalized) or ('?' in normalized and not explicit_request):
        return None
    if not prefix and not allow_short:
        return None
    body = normalized[prefix.end():] if prefix else normalized
    matches = list(_DRINKS.finditer(body))
    if not matches:
        return None
    items = list(dict.fromkeys(_BEVERAGES[int(match.lastgroup[4:])][0] for match in matches))
    remainder = _DRINKS.sub(' ', body)
    qualifiers, remainder = _qualifiers(remainder, unknown_conflicts=len(matches) > 1)
    if qualifiers is None or not _only_filler(remainder):
        return None
    if len(matches) > 1 and qualifiers:
        clauses = [clause for clause in re.split(r'\band\b', body) if _DRINKS.search(clause)]
        trailing_both = bool(re.search(r'\bboth\b', body[matches[-1].end():]))
        for slot, value in tuple(qualifiers.items()):
            per_clause = [_qualifiers(_DRINKS.sub(' ', clause))[0] for clause in clauses]
            if not trailing_both and not all(details and details.get(slot) == value for details in per_clause):
                # A modifier attached to only one drink says nothing about the
                # other drink. Keep the selection; leave that detail unknown.
                qualifiers.pop(slot)
    return {'items': items, **qualifiers}


def _question_slots(text):
    """Classify narrow order-detail questions, not arbitrary words like 'size'."""
    normalized = _normal(text)
    if '?' not in normalized:
        return set()
    slots = set()
    # Choices/details are new information, not a repeated item-selection ask.
    if (re.search(r'\b(?:hot|warm)\s+or\s+(?:iced|cold)\b|\b(?:iced|cold)\s+or\s+(?:hot|warm)\b', normalized)
            or re.search(r'\bwhat (?:drink )?temperature\b', normalized)
            or re.search(r'\b(?:would|do) you (?:like|want|prefer)\b.{0,35}\b(?:hot|iced|cold|warm)\b', normalized)):
        slots.add('temperature')
    if (re.search(r'\b(?:what|which) size(?:\s+(?:would|do|will|should|is|are|for)|\s*[?,.!]|$)', normalized)
            or re.search(r'\b(?:specify|provide|tell|choose)(?:\s+(?:me|us))?\s+(?:the|a|your)\s+size\b', normalized)
            or re.search(r'\bsmall\s*(?:,\s*medium\s*)?(?:,?\s*or\s*)large\b', normalized)
            or re.search(r'\b(?:would|do) you (?:like|want|prefer)\b.{0,30}\b(?:small|medium|large)\b', normalized)):
        slots.add('size')
    if (re.search(r'\bfor here\s+or\s+(?:to go|take away|takeaway)\b', normalized)
            or re.search(r'\b(?:to go|take away|takeaway)\s+or\s+for here\b', normalized)
            or re.search(r'\b(?:would|do) you (?:like|want|prefer)\b.{0,25}\b(?:for here|to go|take away|takeaway)\b', normalized)):
        slots.add('service')
    # Extra items, reasons, quantity, additives and confirmations are distinct
    # facets; their wording may legitimately mention the existing selection.
    distinct = (slots or re.search(r'\b(?:anything else|something else|why|how many|for both of us|for you both|'
                                  r'with milk|without milk|with sugar|without sugar|about|describe)\b', normalized))
    if not distinct and (re.search(r'\bwhat (?:would|do|will) you (?:like|want)(?: to (?:order|drink|have))?\s*(?:today\s*)?[?,.!]', normalized)
            or re.search(r'\bwhat (?:would|do|will) you (?:like|want) to (?:order|drink|have)\b', normalized)
            or re.search(r'\bwhat (?:can|may|could) i (?:get|bring)(?: for)? you\b', normalized)
            or re.search(r'\bwhat (?:is your order|are you (?:ordering|having|drinking))\b', normalized)
            or re.search(r'\bwhich (?:drink|beverage) (?:would|do|will) you (?:like|want|prefer)\b', normalized)):
        slots.add('items')
    # A compound initial question still repeats its answered item-choice part.
    if re.search(r'\bwhat (?:would|do|will) you (?:like|want) to (?:order|drink|have)\s*,?\s+and why\b', normalized):
        slots.add('items')
    return slots


def _detail(text, expected):
    normalized = _normal(text)
    if not expected or _NEGATIVE_OR_UNCERTAIN.search(normalized) or '?' in normalized:
        return None
    body = _DETAIL_PREFIX.sub('', normalized)
    found, remainder = _qualifiers(body)
    if not found or expected not in found or not _only_filler(remainder):
        return None
    return found


def _new_state(values, text, turn):
    return {'frame': 'cafe_order', 'role': 'barista', 'items': values['items'],
            'facts': {slot: {'value': value, 'evidence': text, 'turn': turn} for slot, value in values.items()}}


def cafe_state(request, transcript):
    """Return a supported current food order, or None for ordinary conversation.

    ``facts`` maps items/temperature/size/service to value, exact raw user-turn
    evidence and user-turn index. ``next_action`` names the first missing detail
    (ASK_TEMPERATURE, ASK_SIZE, ASK_SERVICE), else NEXT_CONTRIBUTION. Neither
    assistant statements nor unrelated/ambiguous turns supply order facts.
    """
    if (not isinstance(request, dict) or request.get('topic_id') != 'food' or request.get('reference_text')
            or not isinstance(transcript, str) or not transcript.strip()):
        return None
    history = request.get('history', [])
    if not isinstance(history, list):
        return None
    state, expected, user_turn = None, 'items', 0
    messages = [*history, {'role': 'user', 'content': transcript}]
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            return None
        text = message['content']
        if message.get('role') == 'assistant':
            slots = _question_slots(text)
            expected = next(iter(slots)) if len(slots) == 1 else None
            continue
        if message.get('role') != 'user':
            continue
        user_turn += 1
        if _CANCEL.search(_normal(text)):
            state, expected = None, None
            continue
        selection = _selection(text, allow_short=expected == 'items')
        if selection:
            # A new explicit choice starts a new order; old size/temperature
            # cannot silently carry over to a different selection.
            state = _new_state(selection, text, user_turn)
        elif state:
            detail = _detail(text, expected)
            if detail:
                state['facts'].update({slot: {'value': value, 'evidence': text, 'turn': user_turn}
                                       for slot, value in detail.items()})
            else:
                # An unrecognized continuation may be a topic change or a
                # question. Let Qwen respond freely instead of forcing a slot.
                state = None
        expected = None
    if state is None:
        return None
    state['next_action'] = next((action for slot, action in (
        ('temperature', 'ASK_TEMPERATURE'), ('size', 'ASK_SIZE'), ('service', 'ASK_SERVICE'))
        if slot not in state['facts']), 'NEXT_CONTRIBUTION')
    return state


def state_note(state):
    if not state:
        return ''
    descriptions = {'ASK_TEMPERATURE': 'a temperature', 'ASK_SIZE': 'a size', 'ASK_SERVICE': 'a for-here-or-to-go'}
    action = state.get('next_action', 'NEXT_CONTRIBUTION')
    known = '; '.join(f"{slot}={', '.join(fact['value']) if isinstance(fact['value'], list) else fact['value']}"
                      for slot, fact in state.get('facts', {}).items())
    instruction = (f'You MUST ask ONE short {descriptions[action]} question now.' if action in descriptions
                   else 'Make one brief new contribution; a question is optional.')
    return (f'Active cafe order. You are the barista, not the customer. Known: {known}. '
            f'Next action: {action}. {instruction} '
            'Do not recap the order, welcome again, ask its favourite, or speak as the customer.')


def _known_order_recap(text, state):
    prefix = re.match(r'^(?:i see (?:that )?you (?:want|would like)|you (?:want|would like|ordered|have ordered)|your order is)\b\s*', text)
    if not prefix:
        return False
    body = text[prefix.end():]
    items = [_BEVERAGES[int(match.lastgroup[4:])][0] for match in _DRINKS.finditer(body)]
    details, remainder = _qualifiers(_DRINKS.sub(' ', body))
    if details is None or not (items or details) or not _only_filler(remainder):
        return False
    if not set(items).issubset(state.get('items', [])):
        return False
    facts = state.get('facts', {})
    return all(slot in facts and facts[slot]['value'] == value for slot, value in details.items())


def _unsupported_business_claim(text):
    if '?' in text:
        return None
    if (re.search(r'\b(?:i|we|you) can only (?:provide|serve|order|have|get|buy)\b', text)
            or re.search(r'\bonly (?:one|two|three|\d+) (?:items?|drinks?|orders?) (?:is|are|per|at)\b', text)):
        return 'unsupported_item_limit'
    if (re.search(r'\b(?:we(?: are|\x27re)|i(?: am|\x27m)|(?:our |the )?caf[eé] is)\s+(?:out of|sold out of)\b', text)
            or re.search(r'\bwe (?:do not|don\x27t|cannot|can\x27t) (?:have|serve|offer)\b', text)
            or re.search(r'\b(?:is|are) (?:unavailable|out of stock|in stock)\b', text)):
        return 'unsupported_stock'
    if (re.search(r'[$£€¥]\s*\d', text)
            or re.search(r'\b(?:costs?|prices?|total|comes to)\b.{0,35}\b(?:dollars?|pounds?|euros?|yen|dong)\b', text)
            or re.search(r'\b(?:is|are)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:dollars?|pounds?|euros?)\b', text)):
        return 'unsupported_price'
    return None


def question_violation(text, state):
    """Reject only narrow restarts, customer-role recaps and known-slot reasks."""
    if not state or not isinstance(text, str):
        return None
    normalized = _normal(text)
    business_claim = _unsupported_business_claim(normalized)
    if business_claim:
        return business_claim
    if re.match(r'^\s*welcome\b', normalized):
        return 'cafe_restart'
    if _known_order_recap(normalized, state):
        return 'cafe_order_recap'
    if _selection(text, allow_short=False):
        return 'customer_order_recap'
    if (re.match(r'^(?:both|the|my|our|these|those)\s+drinks?\s+(?:are|should be|will be)\b', normalized)
            and re.search(r'\bplease\b', normalized)):
        details, remainder = _qualifiers(normalized)
        if details and _only_filler(remainder):
            return 'customer_order_recap'
    slots = _question_slots(text)
    for slot in ('items', 'temperature', 'size', 'service'):
        if slot in slots and slot in state.get('facts', {}):
            return 'repeated_' + ('selection' if slot == 'items' else slot)
    return None
