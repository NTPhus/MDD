"""Topic context and grounded per-turn feedback, without loading model weights."""
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock
from http.client import HTTPConnection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdd.demo_chat import ChatError
from mdd.demo import DemoApplication, DemoServer
from mdd.speech_topics import TOPICS, public_topics, guidance
from scripts.local_speech_engine import dialogue_request, follow_up, LocalSpeechEngine
import test_speech_chat as speech_tests
finished = speech_tests.finished


class TopicServiceTests(unittest.TestCase):
    setUp = speech_tests.SpeechChatTests.setUp
    payload = speech_tests.SpeechChatTests.payload
    def topic_payload(self, **extra):
        return {'model_id': 'qwen-base', 'topic_id': 'travel', 'request_id': 'b' * 32, **extra}

    def test_opening_is_immediate_authored_idempotent_and_not_a_model_prediction(self):
        first = self.service.start_topic(self.topic_payload())
        self.assertEqual(first['status'], 'ready')
        self.assertEqual(first['text'], TOPICS['travel']['opening'])
        self.assertEqual(first['metadata']['source'], 'topic_catalog')
        self.assertEqual(first['metadata']['guidance']['keywords'], guidance('travel')['keywords'])
        self.assertEqual(self.service.start_topic(self.topic_payload())['id'], first['id'])
        self.assertEqual(self.calls, [])
        with self.assertRaises(ChatError): self.service.start_topic(self.topic_payload(topic_id='food'))
        with self.assertRaises(ChatError): self.service.voice({'job_id': first['id'], 'locale': 'vi-VN'})

    def test_unknown_topics_and_client_prompt_injection_are_rejected(self):
        for change in ({'topic_id': '../.env'}, {'topic_id': []}, {'opening': 'client injected'}, {'guidance': {'keywords': ['fake']}}, {'model_id': 'invalid'}):
            with self.subTest(change=change), self.assertRaises(ChatError):
                self.service.start_topic(self.topic_payload(**change))
        for change in ({'topic_id': 'unknown'}, {'topic_id': 'travel', 'practice_id': 'idea'}, {'topic': {'role': 'override'}}):
            with self.subTest(change=change), self.assertRaises(ChatError): self.service.submit_speech(self.payload(**change))

    def test_topic_passes_to_worker_and_retains_only_valid_audio_history(self):
        history = [{'role': 'user', 'content': 'I want to visit Japan.'}, {'role': 'assistant', 'content': 'What would you do there?'}]
        final = finished(self.service, self.service.submit_speech(self.payload(topic_id='travel', history=history))['id'])
        self.assertEqual(final['status'], 'ready')
        self.assertEqual(self.calls[0]['topic_id'], 'travel')
        self.assertEqual(self.calls[0]['history'], history)
        self.assertIsNone(self.calls[0]['reference_text'])

    def test_topic_http_route_has_the_same_origin_guard_and_public_config(self):
        server = DemoServer(('127.0.0.1', 0), DemoApplication(self.root, chat_service=self.service))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            def request(origin):
                connection = HTTPConnection('127.0.0.1', server.server_port)
                connection.request('POST', '/api/chat/topic', json.dumps(self.topic_payload()), {'Content-Type': 'application/json', 'Origin': origin})
                response = connection.getresponse(); data = response.read(); connection.close(); return response.status, json.loads(data)
            self.assertEqual(request('https://evil.invalid')[0], 403)
            status, job = request(f'http://127.0.0.1:{server.server_port}')
            self.assertEqual(status, 200); self.assertEqual(job['text'], TOPICS['travel']['opening'])
            self.assertEqual(len(self.service.config()['topics']), 8)
        finally: server.shutdown(); server.server_close(); thread.join(timeout=3)


class TopicDialogueTests(unittest.TestCase):
    def test_prompt_uses_the_selected_topic_and_original_qwen_with_real_history(self):
        history = [{'role': 'user', 'content': 'Japan.'}, {'role': 'assistant', 'content': 'Why Japan?'}]
        request = dialogue_request({'model_id': 'qwen-trained', 'topic_id': 'travel', 'history': history}, 'I enjoy local food.')
        self.assertEqual(request['model_id'], 'qwen-base')
        self.assertIn('Travel & adventure', request['messages'][0]['content'])
        self.assertEqual(request['messages'][-len(history)-1:-1], history)
        self.assertNotIn({'role': 'assistant', 'content': TOPICS['travel']['opening']}, request['messages'])
        self.assertEqual(request['messages'][-1]['content'], 'I enjoy local food.')
        self.assertEqual(len(history), 2)
        self.assertNotIn('A useful next question is:', request['messages'][0]['content'])

    def test_authored_opening_is_context_only_before_the_first_real_exchange(self):
        request = dialogue_request({'model_id': 'qwen-base', 'topic_id': 'travel', 'history': []}, 'Japan.')
        self.assertEqual(request['messages'][-2], {'role': 'assistant', 'content': TOPICS['travel']['opening']})

    def test_short_answer_retains_previous_question_and_recent_learner_details(self):
        history = [{'role': 'user', 'content': 'I want to visit Japan with my sister.'},
                   {'role': 'assistant', 'content': 'What would you both like to do there?'}]
        request = dialogue_request({'model_id': 'qwen-trained', 'topic_id': 'travel', 'history': history}, 'Try local food.')
        self.assertEqual(request['messages'][-3:-1], history)
        self.assertEqual(request['messages'][-1]['content'], 'Try local food.')

    def test_unclear_speech_and_topic_shift_reach_qwen_without_invented_text(self):
        for utterance in ('fravorit... tomorrow...', "Actually, my train is late. I'm worried."):
            with self.subTest(utterance=utterance):
                history = [{'role': 'user', 'content': 'A cup of coffee, please.'},
                           {'role': 'assistant', 'content': 'Would you like anything else?'}]
                request = dialogue_request({'model_id': 'qwen-trained', 'topic_id': 'food', 'history': history}, utterance)
                self.assertEqual(request['messages'][-1], {'role': 'user', 'content': utterance})
                self.assertEqual(request['messages'][-3:-1], history)
                self.assertNotIn(guidance('food', 1)['question'], request['messages'][0]['content'])
                self.assertEqual(request['model_id'], 'qwen-base')

    def test_open_and_closed_model_questions_are_preserved_without_an_authored_follow_up(self):
        hint, source = follow_up('That sounds nice. What would you like to do when you get there?', 'travel', 1)
        self.assertEqual(source, 'qwen'); self.assertIn('try local food', hint['keywords'])
        hint, source = follow_up('That sounds nice. Do you like travelling?', 'travel', 1)
        self.assertEqual(source, 'qwen')
        self.assertEqual(hint['question'], 'Do you like travelling?')
        self.assertEqual(hint['keywords'], [])

    def test_question_extraction_accepts_contractions_compound_and_clarification_questions(self):
        for question in ("What's your favourite dish?", 'Sorry, did you mean coffee?',
                         'What would you like to order, and why is it your favorite?'):
            with self.subTest(question=question):
                hint, source = follow_up('I see. ' + question, 'food', 2)
                self.assertEqual(source, 'qwen')
                self.assertEqual(hint['question'], question)
        hint, _ = follow_up('What would you like to order?', 'food', 2)
        self.assertIn('iced coffee', hint['keywords'])
        self.assertNotIn('local dish', hint['keywords'])

    def test_existing_question_is_not_replaced_with_an_authored_question(self):
        history = [{'role': 'assistant', 'content': 'What would you like to share about Japan?'}]
        hint, source = follow_up('What would you like to share about Japan?', 'travel', 2,
                                 transcript='Japan', history=history)
        self.assertEqual(source, 'qwen')
        self.assertEqual(hint['question'], history[0]['content'])
        self.assertEqual(hint['keywords'], [])

    def test_statement_and_unclear_input_do_not_force_a_question_or_catalog_step(self):
        for topic_id in ('travel', None):
            for transcript in ('Coffee', 'fravorit', 'Yesterday I go to the park'):
                with self.subTest(topic_id=topic_id, transcript=transcript):
                    hint, source = follow_up('A café sounds fun.', topic_id, 2, transcript=transcript)
                    self.assertEqual(source, 'none')
                    self.assertIsNone(hint['question'])
                    self.assertEqual(hint['keywords'], [])
                    self.assertEqual(hint['starter'], '')

    def test_free_conversation_keeps_the_actual_model_question(self):
        hint, source = follow_up('A café sounds fun.', None, 1, transcript='Coffee')
        self.assertEqual(source, 'none')
        self.assertIsNone(hint['question'])
        hint, source = follow_up('How did you discover that café?', None, 2, transcript='I go there often.')
        self.assertEqual(source, 'qwen')
        self.assertEqual(hint['question'], 'How did you discover that café?')

    def test_catalog_hints_follow_the_model_question_not_the_turn_number(self):
        for turns in (0, 1, 2, 19):
            hint, source = follow_up('What would you like to do when you get there?', 'travel', turns)
            self.assertEqual(source, 'qwen')
            self.assertIn('try local food', hint['keywords'])
            self.assertNotIn('budget', hint['keywords'])
        hint, _ = follow_up('Did you mean the train is late?', 'food', 2)
        self.assertEqual(hint['keywords'], [])
        self.assertEqual(hint['starter'], '')

    def test_catalog_is_independent_of_pronunciation_and_can_not_be_mutated_by_a_caller(self):
        topics = public_topics(); topics[0]['guidance']['keywords'].clear()
        self.assertEqual(len(public_topics()[0]['guidance']['keywords']), 4)
        self.assertNotIn('score', json.dumps(topics))

    def test_engine_emits_audio_evidence_before_dialogue_without_appending_a_question(self):
        for reply in ('That sounds exciting.', 'Do you like travelling?',
                      'What would you like to order, and why is it your favorite?'):
            with self.subTest(reply=reply):
                engine = object.__new__(LocalSpeechEngine)
                engine.analyze = Mock(return_value={'transcript': 'I like Japan', 'reference': None, 'comparison': None, 'phonemes': []})
                events = []; model = Mock()
                def generate(request, cancelled, emit):
                    self.assertEqual(request['model_id'], 'qwen-base')
                    self.assertTrue(any(event.get('type') == 'speech' for event in events))
                    emit({'type': 'chunk', 'text': reply}); return {'finish_reason': 'stop'}
                model.generate.side_effect = generate
                metadata = engine.run({'model_id': 'qwen-trained', 'topic_id': 'travel', 'audio_base64': 'eA==', 'history': []}, threading.Event(), events.append, model)
                text = ''.join(event['text'] for event in events if event['type'] == 'chunk')
                self.assertEqual(text, reply)
                final = [event['speech'] for event in events if event['type'] == 'speech'][-1]
                self.assertEqual(final['follow_up'], {'source': 'qwen' if '?' in reply else 'none',
                                                     'question': reply if '?' in reply else None})
                self.assertEqual(final['coaching']['status'], 'unavailable')
                self.assertEqual(final['guidance']['keywords'], [])
                self.assertEqual(metadata['topic_id'], 'travel')


if __name__ == '__main__': unittest.main()
