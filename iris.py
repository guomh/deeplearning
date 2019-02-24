import tensorflow as tf
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
x_vals = np.asarray(iris.data)
y_vals = iris.target
y_ = []
for i in range(len(y_vals)):
	if y_vals[i] == 0:
		y_.append([1,0,0])
	elif y_vals[i] == 1:
		y_.append([0,1,0])
	else:
		y_.append([0,0,1])
y_vals = np.asarray(y_)
y_vals = y_vals.reshape((y_vals.shape[0],3))

tmp_list = []
for (x,y) in zip(x_vals,y_vals):
	tmp_list.append((x,y))
tmp_list = np.asarray(tmp_list)
np.random.shuffle(tmp_list)
x_vals = []
y_vals = []
for i in tmp_list:
	x_vals.append(i[0])
	y_vals.append(i[1])
x_vals = np.asarray(x_vals)
y_vals = np.asarray(y_vals)
print('x.shape',x_vals.shape)
print('y.shape',y_vals.shape)
x_train = x_vals[:100]
x_test = x_vals[100:]
y_train = y_vals[:100]
y_test = y_vals[100:]
# print('x_train',x_train)
print('y_train',y_train.dtype)

batch_size = 25
episode = 5000
learning_rate = 0.001
d1 = 10
d2 = 10
print('x_vals.shape',x_vals.shape)
print('y_vals.shape',y_vals.shape)
x_data = tf.placeholder(shape=[None,4],dtype=tf.float32)
y_data = tf.placeholder(shape=[None,3],dtype=tf.float32)

W1 = tf.Variable(tf.random_normal(shape=[4,d1])*0.01)
B1 = tf.Variable(tf.zeros(shape=[1,d1]))
W2 = tf.Variable(tf.random_normal(shape=[d1,d2])*0.01)
B2 = tf.Variable(tf.zeros(shape=[1,d2]))
W3 = tf.Variable(tf.random_normal(shape=[d2,3])*0.01)
B3 = tf.Variable(tf.zeros(shape=[1,3]))

def calculate(x):
	model_output = tf.add(tf.matmul(x,W1),B1)
	model_output = tf.nn.relu(model_output)
	model_output = tf.add(tf.matmul(model_output,W2),B2)
	model_output = tf.nn.relu(model_output)
	model_output = tf.add(tf.matmul(model_output,W3),B3)
	# model_output = tf.nn.softmax(model_output)
	return model_output

model_output = calculate(x_data)

def test(x,y,result):
	xt = np.asarray(x)
	xt = np.reshape(xt,(1,4))
	x_data = tf.placeholder(shape=[1,4],dtype=tf.float32)
	output = calculate(x_data)
	output = sess.run(output,feed_dict={x_data:xt})
	if sess.run(tf.equal(tf.argmax(output,1),tf.argmax(y))):
		result['yes'] += 1
	else:
		result['no'] += 1
# loss = tf.reduce_mean(tf.square(y_data -tf.to_float(tf.argmax(model_output,1))))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data,logits=model_output))

my_opt = tf.train.AdamOptimizer(learning_rate)
train_step = my_opt.minimize(loss)
prediction = tf.round(tf.nn.softmax(model_output))
predictions_correct = tf.cast(tf.equal(prediction,y_data),tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	lostt_vec = []
	for i in range(episode):
		_,p_loss,_accuracy,_pred = sess.run([train_step,loss,accuracy,prediction],feed_dict={x_data:x_train,y_data:y_train})
		if (i+1) % 25 != 0:
			continue
		print(p_loss,_accuracy)
	
	result = {'yes':0,'no':0}
	for i in range(len(x_test)):
		test(x_test[i],y_test[i],result)
	
	print('yes=',result['yes'],'no=',result['no'],'accuracy=',result['yes']/(result['yes']+result['no']))
