#http://iamtrask.github.io/iamtrask.github.io/page2/
import numpy as np


class SentenceID:
	@staticmethod
	def adapt_sentence(sentence):
		"""Splits sentence's words and removes all chars that are not in the alphabet"""
		alph = [chr(i) for i in range(97, 97 + 26)]
		alph = alph + [letter.upper() for letter in alph] + [" "]
		new_sentence = ""
		for c in sentence:
			if c in alph:
				new_sentence += c
		return new_sentence

	def create_word_ids(self, sentence):
		"""Creates ids for words in sentence"""
		spilt_sentence = self.adapt_sentence(sentence).split(" ")
		utf8_array = [word.encode("utf-8") for word in spilt_sentence]
		word_ids = []
		for array in utf8_array:
			id = []
			for b in array:
				id.append(b)
			word_ids.append(id)
		return word_ids

	@staticmethod
	def concat_arrays(arrays):
		"""concatenate arrays"""
		final_array = []
		for array in arrays:
			final_array += array
		return final_array

	@staticmethod
	def add_padding(current, max_len):
		current_len = len(current)
		if current_len != max_len:
			return current + [0 for _ in range(max_len - current_len)]
		return current

	def encode(self, sentence):
		"""Creates the sentences id based off of letters used. Not positions. As position does not matter"""
		ids = self.create_word_ids(sentence)
		sentence_id = self.concat_arrays(ids)
		test = ""
		for n in sentence_id:
			test += "{0:b}".format(n)
		return test


class Encode:
	def __init__(self):
		self.sentence_id = SentenceID()

	@staticmethod
	def get_max_set_len(training, test):
		return max(
			max([len(encoded_text) for encoded_text in training]),
			max([len(encoded_text) for encoded_text in test]))

	def encode_set(self, current_set):
		return [[int(bit) for bit in self.sentence_id.encode(text).strip()]
										for text in current_set]

	def padding(self, current_set, max_len):
		return [
			self.sentence_id.add_padding(encoded_text, max_len)
			for encoded_text in current_set
		]


encode_data = Encode()


def read_set(file_name, is_int=False):
	current_set = []
	with open(file_name) as input_set:
		for input_i in input_set:
			if is_int:
				current_set.append([int(input_i.strip("\n"))])
			else:
				current_set.append(input_i.strip("\n"))
	return current_set


def initialize_all_data():
	training_input_set = read_set("training_input_dataset.txt")
	training_output_set = read_set("training_output_dataset.txt", is_int=True)
	testing_input_set = read_set("testing_input_dataset.txt")
	testing_output_set = read_set("testing_output_dataset.txt", is_int=True)

	training_input_set = encode_data.encode_set(training_input_set)
	testing_input_set = encode_data.encode_set(testing_input_set)

	max_len = encode_data.get_max_set_len(training_input_set, testing_input_set)

	training_input_set = np.array(
		encode_data.padding(training_input_set, max_len))
	training_output_set = np.array(training_output_set)
	testing_input_set = np.array(encode_data.padding(testing_input_set, max_len))
	testing_output_set = np.array(testing_output_set)

	return training_input_set, training_output_set, testing_input_set, testing_output_set


def nonlin(x, deriv=False):
	if (deriv == True):
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))


training_input_set, training_output_set, testing_input_set, testing_output_set = initialize_all_data(
)

np.random.seed(1)

iterations = 5000
alpha = 1
hidden_size = len(training_input_set)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((len(training_input_set[0]), hidden_size)) - 1
print(syn0)
syn1 = 2 * np.random.random((hidden_size, hidden_size)) - 1
syn2 = 2 * np.random.random((hidden_size, 1)) - 1

l2 = []
for j in range(iterations + 1):
	# Feed forward through layers 0, 1, and 2
	l0 = training_input_set
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))
	l3 = nonlin(np.dot(l2, syn2))

	# how much did we miss the target value?
	l3_error = training_output_set - l3

	if j % int(iterations * 0.2) == 0:
		print(f"{j} Error: {str(np.mean(np.abs(l3_error)))}")

	# in what direction is the target value?
	# were we really sure? if so, don't change too much.
	l3_delta = l3_error * nonlin(l3, deriv=True)

	# how much did each l1 value contribute to the l2 error (according to the weights)?
	l2_error = l3_delta.dot(syn2.T)

	# in what direction is the target l1?
	# were we really sure? if so, don't change too much.
	l2_delta = l3_error * nonlin(l1, deriv=True)
	
	l1_error = l2_delta.dot(syn1.T)

	# in what direction is the target l1?
	# were we really sure? if so, don't change too much.
	l1_delta = l2_error * nonlin(l1, deriv=True)

	syn2 += alpha * l2.T.dot(l3_delta)
	syn1 += alpha * l1.T.dot(l2_delta)
	syn0 += alpha * l0.T.dot(l1_delta)

for testing_input_data, testing_encoded_data, testing_output_data in zip(
		read_set("testing_input_dataset.txt"), testing_input_set,
		testing_output_set):
	print(f"\n{testing_input_data}")

	l0 = testing_encoded_data
  print(testing_encoded_data)
  assert False
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))
	l3 = nonlin(np.dot(l2, syn2))

	predicted = "😃"
	if l3 < 0.5:
		predicted = "☹️"

	actual_sentiment = "😃"
	if testing_output_data == 0:
		actual_sentiment = "☹️"

	print(f"Predicted: {predicted} {l3}")
	print(f"Actual: \t {actual_sentiment} {testing_output_data}")
