"""Authored speaking scenarios, not model predictions or pronunciation labels."""
from copy import deepcopy
import re


def scenario(title, description, icon, level, role, opening, steps):
    return dict(title=title, description=description, icon=icon, level=level, role=role,
                opening=opening, steps=[dict(question=q, keywords=words.split('|'), starter=starter)
                                       for q, words, starter in steps])


TOPICS = {
    'daily-life': scenario('Everyday life', 'Những câu chuyện nhỏ mỗi ngày', 'sun', 'A2', 'a friendly neighbour',
        "Hi! Let's talk about your day. What do you usually enjoy doing in the morning?", [
            ('What do you usually enjoy doing in the morning?', 'wake up|breakfast|coffee|routine', 'I usually start my day with ...'),
            ('How do you like to relax after a busy day?', 'unwind|go for a walk|listen to music|feel relaxed', 'After a busy day, I like to ... because ...'),
            ('What would your ideal weekend look like?', 'sleep in|spend time|meet friends|outdoors', 'My ideal weekend would include ...'),
            ('What is one habit you would like to change, and why?', 'healthy habit|make time|improve|balance', 'I would like to ... because ...')]),
    'travel': scenario('Travel & adventure', 'Lên kế hoạch cho chuyến đi tiếp theo', 'plane', 'A2–B1', 'a curious travel companion',
        "Let's plan a trip together! Where would you love to travel, and what makes that place special to you?", [
            ('Where would you love to travel, and why?', 'destination|visit|beautiful scenery|local culture', "I'd love to visit ... because ..."),
            ('What would you like to do when you get there?', 'explore|sightseeing|try local food|take photos', 'When I get there, I would like to ...'),
            ('How would you plan the trip on a small budget?', 'budget|accommodation|public transport|book ahead', 'To save money, I would ...'),
            ('What have you learned from a memorable trip?', 'memorable|discover|experience|learn', 'One trip I remember was ... because ...')]),
    'food': scenario('Food & cafés', 'Gọi món và chia sẻ món ăn yêu thích', 'cup', 'A2', 'a friendly cafe server',
        "Hi! What would you like to order?", [
            ('What would you like to order?', 'a cup of|iced coffee|milk tea|please', "I'd like ..."),
            ('How would you describe a dish you really enjoy?', 'flavour|crispy|spicy|ingredients', 'My favourite dish is ... It tastes ...'),
            ('What food would you recommend to a visitor to your hometown?', 'local dish|recommend|traditional|fresh', 'You should try ... because ...'),
            ('How do you choose where to eat with friends?', 'atmosphere|affordable|menu|share a meal', 'When choosing a restaurant, I look for ...')]),
    'work': scenario('Work & careers', 'Nói về công việc và mục tiêu của bạn', 'briefcase', 'B1', 'a supportive colleague',
        "It's nice to meet you! What do you do or study, and what do you enjoy most about it?", [
            ('What do you enjoy most about your work or studies?', 'work on|team|interesting|solve problems', 'I work as ... and I enjoy ...'),
            ('How do you handle a difficult task?', 'challenge|ask for help|break it down|collaborate', 'When a task is difficult, I usually ...'),
            ('What skill would you like to develop next, and why?', 'improve|communication|learn|opportunity', 'I would like to develop ... because ...'),
            ('What would your ideal job look like?', 'flexible|meaningful|responsibility|career goal', 'My ideal job would allow me to ...')]),
    'hobbies': scenario('Hobbies & free time', 'Chia sẻ những điều khiến bạn vui', 'spark', 'A2', 'a friendly new acquaintance',
        "I'm looking for something fun to try! What do you love doing in your free time?", [
            ('What do you love doing in your free time?', 'enjoy|in my free time|relax|creative', 'In my free time, I really enjoy ...'),
            ('How did you get interested in that hobby?', 'started|inspired by|try something new|practise', 'I first became interested in ... when ...'),
            ('What would you recommend to someone trying it for the first time?', 'beginner|take it slowly|equipment|have fun', 'For a beginner, I would recommend ...'),
            ('What new hobby would you like to try, and why?', 'curious|learn|challenge myself|outdoors', 'I would love to try ... because ...')]),
    'technology': scenario('Technology & ideas', 'Ứng dụng, AI và cuộc sống số', 'laptop', 'B1', 'a curious friend interested in technology',
        "Technology changes so quickly! What app or device makes your everyday life easier?", [
            ('What app or device makes your everyday life easier?', 'useful|convenient|save time|keep in touch', 'I often use ... because it helps me ...'),
            ('How do you decide whether to try a new app?', 'features|easy to use|privacy|recommendation', 'Before trying an app, I usually ...'),
            ('How could technology help people learn better?', 'artificial intelligence|feedback|personalised|access', 'Technology could help learners by ...'),
            ('What would you change about your digital habits?', 'screen time|take a break|focus|balance', 'I would like to spend less time ... and more time ...')]),
    'learning': scenario('Learning & goals', 'Kể về điều bạn muốn học', 'book', 'A2–B1', 'an encouraging study partner',
        "Let's learn something new together! What is one thing you would love to learn, and why?", [
            ('What is one thing you would love to learn, and why?', 'learn|goal|interested in|improve', 'I would love to learn ... because ...'),
            ('How do you stay motivated when learning is difficult?', 'practise regularly|small steps|progress|motivation', 'When learning feels difficult, I ...'),
            ('What learning method works best for you?', 'hands-on|take notes|watch videos|study together', 'I learn best when I ...'),
            ('How would you use your new skill in everyday life?', 'confidence|opportunity|share|put into practice', 'With this skill, I could ...')]),
    'city': scenario('Your city', 'Giới thiệu nơi bạn sống', 'city', 'A2', 'a visitor getting to know the learner\'s hometown',
        "I'd love to explore your hometown! What place should I visit first, and what makes it special?", [
            ('What place should I visit first in your hometown, and why?', 'hometown|recommend|famous for|nearby', 'You should visit ... because ...'),
            ('How would you spend a perfect day in your city?', 'walk around|market|park|local food', 'A perfect day here would start with ...'),
            ('What do you like most about your neighbourhood?', 'friendly|peaceful|convenient|community', 'What I like most about my neighbourhood is ...'),
            ('What would you improve about your city?', 'public transport|green space|traffic|cleaner', 'If I could change one thing, I would ...')]),
}


def guidance(topic_id, turns=0, question=None):
    topic = TOPICS[topic_id]
    steps = topic['steps']
    index = turns % len(steps)
    if question:
        tokens = set(re.findall(r'[a-z]+', question.lower())) - {'what', 'how', 'would', 'you', 'your', 'the', 'a', 'and', 'to', 'do', 'is', 'in', 'of'}
        scores = [len(tokens & set(re.findall(r'[a-z]+', step['question'].lower()))) for step in steps]
        if max(scores, default=0) > 1:
            index = max(range(len(steps)), key=lambda i: scores[i])
    return {'topic_id': topic_id, 'title': topic['title'], 'source': 'topic_catalog',
            **deepcopy(steps[index]), **({'question': question} if question else {})}


def public_topics():
    return [{'id': key, **{name: topic[name] for name in ('title', 'description', 'icon', 'level', 'opening')},
             'guidance': guidance(key)} for key, topic in TOPICS.items()]
