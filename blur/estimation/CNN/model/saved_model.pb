Ε2
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878σΤ&

conv01/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv01/kernel
x
!conv01/kernel/Read/ReadVariableOpReadVariableOpconv01/kernel*'
_output_shapes
:H*
dtype0
o
conv01/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv01/bias
h
conv01/bias/Read/ReadVariableOpReadVariableOpconv01/bias*
_output_shapes	
:*
dtype0

block0/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock0/conv1/kernel

'block0/conv1/kernel/Read/ReadVariableOpReadVariableOpblock0/conv1/kernel*(
_output_shapes
:*
dtype0
{
block0/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock0/conv1/bias
t
%block0/conv1/bias/Read/ReadVariableOpReadVariableOpblock0/conv1/bias*
_output_shapes	
:*
dtype0

block0/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock0/conv2/kernel

'block0/conv2/kernel/Read/ReadVariableOpReadVariableOpblock0/conv2/kernel*(
_output_shapes
:*
dtype0
{
block0/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock0/conv2/bias
t
%block0/conv2/bias/Read/ReadVariableOpReadVariableOpblock0/conv2/bias*
_output_shapes	
:*
dtype0

block1/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock1/conv1/kernel

'block1/conv1/kernel/Read/ReadVariableOpReadVariableOpblock1/conv1/kernel*(
_output_shapes
:*
dtype0
{
block1/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock1/conv1/bias
t
%block1/conv1/bias/Read/ReadVariableOpReadVariableOpblock1/conv1/bias*
_output_shapes	
:*
dtype0

block1/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock1/conv2/kernel

'block1/conv2/kernel/Read/ReadVariableOpReadVariableOpblock1/conv2/kernel*(
_output_shapes
:*
dtype0
{
block1/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock1/conv2/bias
t
%block1/conv2/bias/Read/ReadVariableOpReadVariableOpblock1/conv2/bias*
_output_shapes	
:*
dtype0

block2/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2/conv1/kernel

'block2/conv1/kernel/Read/ReadVariableOpReadVariableOpblock2/conv1/kernel*(
_output_shapes
:*
dtype0
{
block2/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2/conv1/bias
t
%block2/conv1/bias/Read/ReadVariableOpReadVariableOpblock2/conv1/bias*
_output_shapes	
:*
dtype0

block2/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2/conv2/kernel

'block2/conv2/kernel/Read/ReadVariableOpReadVariableOpblock2/conv2/kernel*(
_output_shapes
:*
dtype0
{
block2/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2/conv2/bias
t
%block2/conv2/bias/Read/ReadVariableOpReadVariableOpblock2/conv2/bias*
_output_shapes	
:*
dtype0

block2/convshortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock2/convshortcut/kernel

.block2/convshortcut/kernel/Read/ReadVariableOpReadVariableOpblock2/convshortcut/kernel*(
_output_shapes
:*
dtype0

block2/convshortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock2/convshortcut/bias

,block2/convshortcut/bias/Read/ReadVariableOpReadVariableOpblock2/convshortcut/bias*
_output_shapes	
:*
dtype0

block3/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3/conv1/kernel

'block3/conv1/kernel/Read/ReadVariableOpReadVariableOpblock3/conv1/kernel*(
_output_shapes
:*
dtype0
{
block3/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3/conv1/bias
t
%block3/conv1/bias/Read/ReadVariableOpReadVariableOpblock3/conv1/bias*
_output_shapes	
:*
dtype0

block3/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3/conv2/kernel

'block3/conv2/kernel/Read/ReadVariableOpReadVariableOpblock3/conv2/kernel*(
_output_shapes
:*
dtype0
{
block3/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3/conv2/bias
t
%block3/conv2/bias/Read/ReadVariableOpReadVariableOpblock3/conv2/bias*
_output_shapes	
:*
dtype0

block3/convshortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock3/convshortcut/kernel

.block3/convshortcut/kernel/Read/ReadVariableOpReadVariableOpblock3/convshortcut/kernel*(
_output_shapes
:*
dtype0

block3/convshortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock3/convshortcut/bias

,block3/convshortcut/bias/Read/ReadVariableOpReadVariableOpblock3/convshortcut/bias*
_output_shapes	
:*
dtype0

block4/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4/conv1/kernel

'block4/conv1/kernel/Read/ReadVariableOpReadVariableOpblock4/conv1/kernel*(
_output_shapes
:*
dtype0
{
block4/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4/conv1/bias
t
%block4/conv1/bias/Read/ReadVariableOpReadVariableOpblock4/conv1/bias*
_output_shapes	
:*
dtype0

block4/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4/conv2/kernel

'block4/conv2/kernel/Read/ReadVariableOpReadVariableOpblock4/conv2/kernel*(
_output_shapes
:*
dtype0
{
block4/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4/conv2/bias
t
%block4/conv2/bias/Read/ReadVariableOpReadVariableOpblock4/conv2/bias*
_output_shapes	
:*
dtype0

block4/convshortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock4/convshortcut/kernel

.block4/convshortcut/kernel/Read/ReadVariableOpReadVariableOpblock4/convshortcut/kernel*(
_output_shapes
:*
dtype0

block4/convshortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock4/convshortcut/bias

,block4/convshortcut/bias/Read/ReadVariableOpReadVariableOpblock4/convshortcut/bias*
_output_shapes	
:*
dtype0

block5/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5/conv1/kernel

'block5/conv1/kernel/Read/ReadVariableOpReadVariableOpblock5/conv1/kernel*(
_output_shapes
:*
dtype0
{
block5/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5/conv1/bias
t
%block5/conv1/bias/Read/ReadVariableOpReadVariableOpblock5/conv1/bias*
_output_shapes	
:*
dtype0

block5/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5/conv2/kernel

'block5/conv2/kernel/Read/ReadVariableOpReadVariableOpblock5/conv2/kernel*(
_output_shapes
:*
dtype0
{
block5/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5/conv2/bias
t
%block5/conv2/bias/Read/ReadVariableOpReadVariableOpblock5/conv2/bias*
_output_shapes	
:*
dtype0

block5/convshortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock5/convshortcut/kernel

.block5/convshortcut/kernel/Read/ReadVariableOpReadVariableOpblock5/convshortcut/kernel*(
_output_shapes
:*
dtype0

block5/convshortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock5/convshortcut/bias

,block5/convshortcut/bias/Read/ReadVariableOpReadVariableOpblock5/convshortcut/bias*
_output_shapes	
:*
dtype0

block6/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock6/conv1/kernel

'block6/conv1/kernel/Read/ReadVariableOpReadVariableOpblock6/conv1/kernel*(
_output_shapes
:*
dtype0
{
block6/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock6/conv1/bias
t
%block6/conv1/bias/Read/ReadVariableOpReadVariableOpblock6/conv1/bias*
_output_shapes	
:*
dtype0

block6/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock6/conv2/kernel

'block6/conv2/kernel/Read/ReadVariableOpReadVariableOpblock6/conv2/kernel*(
_output_shapes
:*
dtype0
{
block6/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock6/conv2/bias
t
%block6/conv2/bias/Read/ReadVariableOpReadVariableOpblock6/conv2/bias*
_output_shapes	
:*
dtype0

block6/convshortcut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock6/convshortcut/kernel

.block6/convshortcut/kernel/Read/ReadVariableOpReadVariableOpblock6/convshortcut/kernel*(
_output_shapes
:*
dtype0

block6/convshortcut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock6/convshortcut/bias

,block6/convshortcut/bias/Read/ReadVariableOpReadVariableOpblock6/convshortcut/bias*
_output_shapes	
:*
dtype0
r

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
fc1/kernel
k
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel* 
_output_shapes
:
*
dtype0
i
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc1/bias
b
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes	
:*
dtype0
r

fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
fc2/kernel
k
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel* 
_output_shapes
:
*
dtype0
i
fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc2/bias
b
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
_output_shapes	
:*
dtype0
r

fc3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
fc3/kernel
k
fc3/kernel/Read/ReadVariableOpReadVariableOp
fc3/kernel* 
_output_shapes
:
*
dtype0
i
fc3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc3/bias
b
fc3/bias/Read/ReadVariableOpReadVariableOpfc3/bias*
_output_shapes	
:*
dtype0
w
fc_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namefc_out/kernel
p
!fc_out/kernel/Read/ReadVariableOpReadVariableOpfc_out/kernel*
_output_shapes
:	*
dtype0
n
fc_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefc_out/bias
g
fc_out/bias/Read/ReadVariableOpReadVariableOpfc_out/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv01/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/conv01/kernel/m

(Adam/conv01/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv01/kernel/m*'
_output_shapes
:H*
dtype0
}
Adam/conv01/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv01/bias/m
v
&Adam/conv01/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv01/bias/m*
_output_shapes	
:*
dtype0

Adam/block0/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block0/conv1/kernel/m

.Adam/block0/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block0/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block0/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block0/conv1/bias/m

,Adam/block0/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block0/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block0/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block0/conv2/kernel/m

.Adam/block0/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block0/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block0/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block0/conv2/bias/m

,Adam/block0/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block0/conv2/bias/m*
_output_shapes	
:*
dtype0

Adam/block1/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block1/conv1/kernel/m

.Adam/block1/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block1/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block1/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block1/conv1/bias/m

,Adam/block1/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block1/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block1/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block1/conv2/kernel/m

.Adam/block1/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block1/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block1/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block1/conv2/bias/m

,Adam/block1/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block1/conv2/bias/m*
_output_shapes	
:*
dtype0

Adam/block2/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block2/conv1/kernel/m

.Adam/block2/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block2/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block2/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block2/conv1/bias/m

,Adam/block2/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block2/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block2/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block2/conv2/kernel/m

.Adam/block2/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block2/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block2/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block2/conv2/bias/m

,Adam/block2/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block2/conv2/bias/m*
_output_shapes	
:*
dtype0
¨
!Adam/block2/convshortcut/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block2/convshortcut/kernel/m
‘
5Adam/block2/convshortcut/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/block2/convshortcut/kernel/m*(
_output_shapes
:*
dtype0

Adam/block2/convshortcut/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block2/convshortcut/bias/m

3Adam/block2/convshortcut/bias/m/Read/ReadVariableOpReadVariableOpAdam/block2/convshortcut/bias/m*
_output_shapes	
:*
dtype0

Adam/block3/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block3/conv1/kernel/m

.Adam/block3/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block3/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block3/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block3/conv1/bias/m

,Adam/block3/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block3/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block3/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block3/conv2/kernel/m

.Adam/block3/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block3/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block3/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block3/conv2/bias/m

,Adam/block3/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block3/conv2/bias/m*
_output_shapes	
:*
dtype0
¨
!Adam/block3/convshortcut/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block3/convshortcut/kernel/m
‘
5Adam/block3/convshortcut/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/block3/convshortcut/kernel/m*(
_output_shapes
:*
dtype0

Adam/block3/convshortcut/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block3/convshortcut/bias/m

3Adam/block3/convshortcut/bias/m/Read/ReadVariableOpReadVariableOpAdam/block3/convshortcut/bias/m*
_output_shapes	
:*
dtype0

Adam/block4/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block4/conv1/kernel/m

.Adam/block4/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block4/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block4/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block4/conv1/bias/m

,Adam/block4/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block4/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block4/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block4/conv2/kernel/m

.Adam/block4/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block4/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block4/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block4/conv2/bias/m

,Adam/block4/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block4/conv2/bias/m*
_output_shapes	
:*
dtype0
¨
!Adam/block4/convshortcut/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block4/convshortcut/kernel/m
‘
5Adam/block4/convshortcut/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/block4/convshortcut/kernel/m*(
_output_shapes
:*
dtype0

Adam/block4/convshortcut/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block4/convshortcut/bias/m

3Adam/block4/convshortcut/bias/m/Read/ReadVariableOpReadVariableOpAdam/block4/convshortcut/bias/m*
_output_shapes	
:*
dtype0

Adam/block5/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block5/conv1/kernel/m

.Adam/block5/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block5/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block5/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block5/conv1/bias/m

,Adam/block5/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block5/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block5/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block5/conv2/kernel/m

.Adam/block5/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block5/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block5/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block5/conv2/bias/m

,Adam/block5/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block5/conv2/bias/m*
_output_shapes	
:*
dtype0
¨
!Adam/block5/convshortcut/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block5/convshortcut/kernel/m
‘
5Adam/block5/convshortcut/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/block5/convshortcut/kernel/m*(
_output_shapes
:*
dtype0

Adam/block5/convshortcut/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block5/convshortcut/bias/m

3Adam/block5/convshortcut/bias/m/Read/ReadVariableOpReadVariableOpAdam/block5/convshortcut/bias/m*
_output_shapes	
:*
dtype0

Adam/block6/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block6/conv1/kernel/m

.Adam/block6/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block6/conv1/kernel/m*(
_output_shapes
:*
dtype0

Adam/block6/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block6/conv1/bias/m

,Adam/block6/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/block6/conv1/bias/m*
_output_shapes	
:*
dtype0

Adam/block6/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block6/conv2/kernel/m

.Adam/block6/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/block6/conv2/kernel/m*(
_output_shapes
:*
dtype0

Adam/block6/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block6/conv2/bias/m

,Adam/block6/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/block6/conv2/bias/m*
_output_shapes	
:*
dtype0
¨
!Adam/block6/convshortcut/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block6/convshortcut/kernel/m
‘
5Adam/block6/convshortcut/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/block6/convshortcut/kernel/m*(
_output_shapes
:*
dtype0

Adam/block6/convshortcut/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block6/convshortcut/bias/m

3Adam/block6/convshortcut/bias/m/Read/ReadVariableOpReadVariableOpAdam/block6/convshortcut/bias/m*
_output_shapes	
:*
dtype0

Adam/fc1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc1/kernel/m
y
%Adam/fc1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc1/kernel/m* 
_output_shapes
:
*
dtype0
w
Adam/fc1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc1/bias/m
p
#Adam/fc1/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc1/bias/m*
_output_shapes	
:*
dtype0

Adam/fc2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc2/kernel/m
y
%Adam/fc2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc2/kernel/m* 
_output_shapes
:
*
dtype0
w
Adam/fc2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc2/bias/m
p
#Adam/fc2/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc2/bias/m*
_output_shapes	
:*
dtype0

Adam/fc3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc3/kernel/m
y
%Adam/fc3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc3/kernel/m* 
_output_shapes
:
*
dtype0
w
Adam/fc3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc3/bias/m
p
#Adam/fc3/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc3/bias/m*
_output_shapes	
:*
dtype0

Adam/fc_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/fc_out/kernel/m
~
(Adam/fc_out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc_out/kernel/m*
_output_shapes
:	*
dtype0
|
Adam/fc_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fc_out/bias/m
u
&Adam/fc_out/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc_out/bias/m*
_output_shapes
:*
dtype0

Adam/conv01/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/conv01/kernel/v

(Adam/conv01/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv01/kernel/v*'
_output_shapes
:H*
dtype0
}
Adam/conv01/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv01/bias/v
v
&Adam/conv01/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv01/bias/v*
_output_shapes	
:*
dtype0

Adam/block0/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block0/conv1/kernel/v

.Adam/block0/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block0/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block0/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block0/conv1/bias/v

,Adam/block0/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block0/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block0/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block0/conv2/kernel/v

.Adam/block0/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block0/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block0/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block0/conv2/bias/v

,Adam/block0/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block0/conv2/bias/v*
_output_shapes	
:*
dtype0

Adam/block1/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block1/conv1/kernel/v

.Adam/block1/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block1/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block1/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block1/conv1/bias/v

,Adam/block1/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block1/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block1/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block1/conv2/kernel/v

.Adam/block1/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block1/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block1/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block1/conv2/bias/v

,Adam/block1/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block1/conv2/bias/v*
_output_shapes	
:*
dtype0

Adam/block2/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block2/conv1/kernel/v

.Adam/block2/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block2/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block2/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block2/conv1/bias/v

,Adam/block2/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block2/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block2/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block2/conv2/kernel/v

.Adam/block2/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block2/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block2/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block2/conv2/bias/v

,Adam/block2/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block2/conv2/bias/v*
_output_shapes	
:*
dtype0
¨
!Adam/block2/convshortcut/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block2/convshortcut/kernel/v
‘
5Adam/block2/convshortcut/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/block2/convshortcut/kernel/v*(
_output_shapes
:*
dtype0

Adam/block2/convshortcut/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block2/convshortcut/bias/v

3Adam/block2/convshortcut/bias/v/Read/ReadVariableOpReadVariableOpAdam/block2/convshortcut/bias/v*
_output_shapes	
:*
dtype0

Adam/block3/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block3/conv1/kernel/v

.Adam/block3/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block3/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block3/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block3/conv1/bias/v

,Adam/block3/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block3/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block3/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block3/conv2/kernel/v

.Adam/block3/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block3/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block3/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block3/conv2/bias/v

,Adam/block3/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block3/conv2/bias/v*
_output_shapes	
:*
dtype0
¨
!Adam/block3/convshortcut/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block3/convshortcut/kernel/v
‘
5Adam/block3/convshortcut/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/block3/convshortcut/kernel/v*(
_output_shapes
:*
dtype0

Adam/block3/convshortcut/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block3/convshortcut/bias/v

3Adam/block3/convshortcut/bias/v/Read/ReadVariableOpReadVariableOpAdam/block3/convshortcut/bias/v*
_output_shapes	
:*
dtype0

Adam/block4/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block4/conv1/kernel/v

.Adam/block4/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block4/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block4/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block4/conv1/bias/v

,Adam/block4/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block4/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block4/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block4/conv2/kernel/v

.Adam/block4/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block4/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block4/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block4/conv2/bias/v

,Adam/block4/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block4/conv2/bias/v*
_output_shapes	
:*
dtype0
¨
!Adam/block4/convshortcut/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block4/convshortcut/kernel/v
‘
5Adam/block4/convshortcut/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/block4/convshortcut/kernel/v*(
_output_shapes
:*
dtype0

Adam/block4/convshortcut/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block4/convshortcut/bias/v

3Adam/block4/convshortcut/bias/v/Read/ReadVariableOpReadVariableOpAdam/block4/convshortcut/bias/v*
_output_shapes	
:*
dtype0

Adam/block5/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block5/conv1/kernel/v

.Adam/block5/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block5/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block5/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block5/conv1/bias/v

,Adam/block5/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block5/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block5/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block5/conv2/kernel/v

.Adam/block5/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block5/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block5/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block5/conv2/bias/v

,Adam/block5/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block5/conv2/bias/v*
_output_shapes	
:*
dtype0
¨
!Adam/block5/convshortcut/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block5/convshortcut/kernel/v
‘
5Adam/block5/convshortcut/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/block5/convshortcut/kernel/v*(
_output_shapes
:*
dtype0

Adam/block5/convshortcut/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block5/convshortcut/bias/v

3Adam/block5/convshortcut/bias/v/Read/ReadVariableOpReadVariableOpAdam/block5/convshortcut/bias/v*
_output_shapes	
:*
dtype0

Adam/block6/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block6/conv1/kernel/v

.Adam/block6/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block6/conv1/kernel/v*(
_output_shapes
:*
dtype0

Adam/block6/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block6/conv1/bias/v

,Adam/block6/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/block6/conv1/bias/v*
_output_shapes	
:*
dtype0

Adam/block6/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/block6/conv2/kernel/v

.Adam/block6/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/block6/conv2/kernel/v*(
_output_shapes
:*
dtype0

Adam/block6/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/block6/conv2/bias/v

,Adam/block6/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/block6/conv2/bias/v*
_output_shapes	
:*
dtype0
¨
!Adam/block6/convshortcut/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/block6/convshortcut/kernel/v
‘
5Adam/block6/convshortcut/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/block6/convshortcut/kernel/v*(
_output_shapes
:*
dtype0

Adam/block6/convshortcut/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/block6/convshortcut/bias/v

3Adam/block6/convshortcut/bias/v/Read/ReadVariableOpReadVariableOpAdam/block6/convshortcut/bias/v*
_output_shapes	
:*
dtype0

Adam/fc1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc1/kernel/v
y
%Adam/fc1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc1/kernel/v* 
_output_shapes
:
*
dtype0
w
Adam/fc1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc1/bias/v
p
#Adam/fc1/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc1/bias/v*
_output_shapes	
:*
dtype0

Adam/fc2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc2/kernel/v
y
%Adam/fc2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc2/kernel/v* 
_output_shapes
:
*
dtype0
w
Adam/fc2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc2/bias/v
p
#Adam/fc2/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc2/bias/v*
_output_shapes	
:*
dtype0

Adam/fc3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/fc3/kernel/v
y
%Adam/fc3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc3/kernel/v* 
_output_shapes
:
*
dtype0
w
Adam/fc3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/fc3/bias/v
p
#Adam/fc3/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc3/bias/v*
_output_shapes	
:*
dtype0

Adam/fc_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/fc_out/kernel/v
~
(Adam/fc_out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc_out/kernel/v*
_output_shapes
:	*
dtype0
|
Adam/fc_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fc_out/bias/v
u
&Adam/fc_out/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc_out/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
»
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ΗΊ
valueΌΊBΈΊ B°Ί

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-8
layer-23
layer-24
layer_with_weights-9
layer-25
layer_with_weights-10
layer-26
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-11
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer_with_weights-13
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-14
(layer-39
)layer-40
*layer_with_weights-15
*layer-41
+layer_with_weights-16
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer_with_weights-17
0layer-47
1layer-48
2layer_with_weights-18
2layer-49
3layer_with_weights-19
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer_with_weights-20
8layer-55
9layer_with_weights-21
9layer-56
:layer_with_weights-22
:layer-57
;layer_with_weights-23
;layer-58
<	optimizer
=trainable_variables
>regularization_losses
?	variables
@	keras_api
A
signatures
 
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
h

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
R
\	variables
]trainable_variables
^regularization_losses
_	keras_api
R
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
R
n	variables
otrainable_variables
pregularization_losses
q	keras_api
h

rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
R
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
R
|	variables
}trainable_variables
~regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
 regularization_losses
‘	keras_api
V
’	variables
£trainable_variables
€regularization_losses
₯	keras_api
V
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
n
ͺkernel
	«bias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
V
°	variables
±trainable_variables
²regularization_losses
³	keras_api
n
΄kernel
	΅bias
Ά	variables
·trainable_variables
Έregularization_losses
Ή	keras_api
n
Ίkernel
	»bias
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
V
ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
V
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
V
Θ	variables
Ιtrainable_variables
Κregularization_losses
Λ	keras_api
V
Μ	variables
Νtrainable_variables
Ξregularization_losses
Ο	keras_api
n
Πkernel
	Ρbias
?	variables
Σtrainable_variables
Τregularization_losses
Υ	keras_api
V
Φ	variables
Χtrainable_variables
Ψregularization_losses
Ω	keras_api
n
Ϊkernel
	Ϋbias
ά	variables
έtrainable_variables
ήregularization_losses
ί	keras_api
n
ΰkernel
	αbias
β	variables
γtrainable_variables
δregularization_losses
ε	keras_api
V
ζ	variables
ηtrainable_variables
θregularization_losses
ι	keras_api
V
κ	variables
λtrainable_variables
μregularization_losses
ν	keras_api
V
ξ	variables
οtrainable_variables
πregularization_losses
ρ	keras_api
V
ς	variables
σtrainable_variables
τregularization_losses
υ	keras_api
n
φkernel
	χbias
ψ	variables
ωtrainable_variables
ϊregularization_losses
ϋ	keras_api
V
ό	variables
ύtrainable_variables
ώregularization_losses
?	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
 regularization_losses
‘	keras_api
V
’	variables
£trainable_variables
€regularization_losses
₯	keras_api
n
¦kernel
	§bias
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
n
¬kernel
	­bias
?	variables
―trainable_variables
°regularization_losses
±	keras_api
V
²	variables
³trainable_variables
΄regularization_losses
΅	keras_api
V
Ά	variables
·trainable_variables
Έregularization_losses
Ή	keras_api
V
Ί	variables
»trainable_variables
Όregularization_losses
½	keras_api
V
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
n
Βkernel
	Γbias
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
n
Θkernel
	Ιbias
Κ	variables
Λtrainable_variables
Μregularization_losses
Ν	keras_api
n
Ξkernel
	Οbias
Π	variables
Ρtrainable_variables
?regularization_losses
Σ	keras_api
n
Τkernel
	Υbias
Φ	variables
Χtrainable_variables
Ψregularization_losses
Ω	keras_api
Ρ
	Ϊiter
Ϋbeta_1
άbeta_2

έdecay
ήlearning_rateBmCmLmMmVmWmhmimrmsm	m	m	m	m	m	m 	ͺm‘	«m’	΄m£	΅m€	Ίm₯	»m¦	Πm§	Ρm¨	Ϊm©	Ϋmͺ	ΰm«	αm¬	φm­	χm?	m―	m°	m±	m²	m³	m΄	¦m΅	§mΆ	¬m·	­mΈ	ΒmΉ	ΓmΊ	Θm»	ΙmΌ	Ξm½	ΟmΎ	ΤmΏ	ΥmΐBvΑCvΒLvΓMvΔVvΕWvΖhvΗivΘrvΙsvΚ	vΛ	vΜ	vΝ	vΞ	vΟ	vΠ	ͺvΡ	«v?	΄vΣ	΅vΤ	ΊvΥ	»vΦ	ΠvΧ	ΡvΨ	ΪvΩ	ΫvΪ	ΰvΫ	αvά	φvέ	χvή	vί	vΰ	vα	vβ	vγ	vδ	¦vε	§vζ	¬vη	­vθ	Βvι	Γvκ	Θvλ	Ιvμ	Ξvν	Οvξ	Τvο	Υvπ

B0
C1
L2
M3
V4
W5
h6
i7
r8
s9
10
11
12
13
14
15
ͺ16
«17
΄18
΅19
Ί20
»21
Π22
Ρ23
Ϊ24
Ϋ25
ΰ26
α27
φ28
χ29
30
31
32
33
34
35
¦36
§37
¬38
­39
Β40
Γ41
Θ42
Ι43
Ξ44
Ο45
Τ46
Υ47
 

B0
C1
L2
M3
V4
W5
h6
i7
r8
s9
10
11
12
13
14
15
ͺ16
«17
΄18
΅19
Ί20
»21
Π22
Ρ23
Ϊ24
Ϋ25
ΰ26
α27
φ28
χ29
30
31
32
33
34
35
¦36
§37
¬38
­39
Β40
Γ41
Θ42
Ι43
Ξ44
Ο45
Τ46
Υ47
²
=trainable_variables
ίlayer_metrics
ΰmetrics
αlayers
 βlayer_regularization_losses
>regularization_losses
γnon_trainable_variables
?	variables
 
YW
VARIABLE_VALUEconv01/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv01/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
²
D	variables
Etrainable_variables
δlayer_metrics
εmetrics
ζlayers
 ηlayer_regularization_losses
Fregularization_losses
θnon_trainable_variables
 
 
 
²
H	variables
Itrainable_variables
ιlayer_metrics
κmetrics
λlayers
 μlayer_regularization_losses
Jregularization_losses
νnon_trainable_variables
_]
VARIABLE_VALUEblock0/conv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock0/conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
²
N	variables
Otrainable_variables
ξlayer_metrics
οmetrics
πlayers
 ρlayer_regularization_losses
Pregularization_losses
ςnon_trainable_variables
 
 
 
²
R	variables
Strainable_variables
σlayer_metrics
τmetrics
υlayers
 φlayer_regularization_losses
Tregularization_losses
χnon_trainable_variables
_]
VARIABLE_VALUEblock0/conv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock0/conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

V0
W1
 
²
X	variables
Ytrainable_variables
ψlayer_metrics
ωmetrics
ϊlayers
 ϋlayer_regularization_losses
Zregularization_losses
όnon_trainable_variables
 
 
 
²
\	variables
]trainable_variables
ύlayer_metrics
ώmetrics
?layers
 layer_regularization_losses
^regularization_losses
non_trainable_variables
 
 
 
²
`	variables
atrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
bregularization_losses
non_trainable_variables
 
 
 
²
d	variables
etrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
fregularization_losses
non_trainable_variables
_]
VARIABLE_VALUEblock1/conv1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1/conv1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
²
j	variables
ktrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
lregularization_losses
non_trainable_variables
 
 
 
²
n	variables
otrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
pregularization_losses
non_trainable_variables
_]
VARIABLE_VALUEblock1/conv2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1/conv2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

r0
s1
 
²
t	variables
utrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
vregularization_losses
non_trainable_variables
 
 
 
²
x	variables
ytrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
zregularization_losses
non_trainable_variables
 
 
 
²
|	variables
}trainable_variables
 layer_metrics
‘metrics
’layers
 £layer_regularization_losses
~regularization_losses
€non_trainable_variables
 
 
 
΅
	variables
trainable_variables
₯layer_metrics
¦metrics
§layers
 ¨layer_regularization_losses
regularization_losses
©non_trainable_variables
_]
VARIABLE_VALUEblock2/conv1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2/conv1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
ͺlayer_metrics
«metrics
¬layers
 ­layer_regularization_losses
regularization_losses
?non_trainable_variables
 
 
 
΅
	variables
trainable_variables
―layer_metrics
°metrics
±layers
 ²layer_regularization_losses
regularization_losses
³non_trainable_variables
_]
VARIABLE_VALUEblock2/conv2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2/conv2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
΄layer_metrics
΅metrics
Άlayers
 ·layer_regularization_losses
regularization_losses
Έnon_trainable_variables
fd
VARIABLE_VALUEblock2/convshortcut/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEblock2/convshortcut/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
Ήlayer_metrics
Ίmetrics
»layers
 Όlayer_regularization_losses
regularization_losses
½non_trainable_variables
 
 
 
΅
	variables
trainable_variables
Ύlayer_metrics
Ώmetrics
ΐlayers
 Αlayer_regularization_losses
regularization_losses
Βnon_trainable_variables
 
 
 
΅
	variables
trainable_variables
Γlayer_metrics
Δmetrics
Εlayers
 Ζlayer_regularization_losses
 regularization_losses
Ηnon_trainable_variables
 
 
 
΅
’	variables
£trainable_variables
Θlayer_metrics
Ιmetrics
Κlayers
 Λlayer_regularization_losses
€regularization_losses
Μnon_trainable_variables
 
 
 
΅
¦	variables
§trainable_variables
Νlayer_metrics
Ξmetrics
Οlayers
 Πlayer_regularization_losses
¨regularization_losses
Ρnon_trainable_variables
_]
VARIABLE_VALUEblock3/conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3/conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

ͺ0
«1

ͺ0
«1
 
΅
¬	variables
­trainable_variables
?layer_metrics
Σmetrics
Τlayers
 Υlayer_regularization_losses
?regularization_losses
Φnon_trainable_variables
 
 
 
΅
°	variables
±trainable_variables
Χlayer_metrics
Ψmetrics
Ωlayers
 Ϊlayer_regularization_losses
²regularization_losses
Ϋnon_trainable_variables
_]
VARIABLE_VALUEblock3/conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3/conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

΄0
΅1

΄0
΅1
 
΅
Ά	variables
·trainable_variables
άlayer_metrics
έmetrics
ήlayers
 ίlayer_regularization_losses
Έregularization_losses
ΰnon_trainable_variables
ge
VARIABLE_VALUEblock3/convshortcut/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEblock3/convshortcut/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Ί0
»1

Ί0
»1
 
΅
Ό	variables
½trainable_variables
αlayer_metrics
βmetrics
γlayers
 δlayer_regularization_losses
Ύregularization_losses
εnon_trainable_variables
 
 
 
΅
ΐ	variables
Αtrainable_variables
ζlayer_metrics
ηmetrics
θlayers
 ιlayer_regularization_losses
Βregularization_losses
κnon_trainable_variables
 
 
 
΅
Δ	variables
Εtrainable_variables
λlayer_metrics
μmetrics
νlayers
 ξlayer_regularization_losses
Ζregularization_losses
οnon_trainable_variables
 
 
 
΅
Θ	variables
Ιtrainable_variables
πlayer_metrics
ρmetrics
ςlayers
 σlayer_regularization_losses
Κregularization_losses
τnon_trainable_variables
 
 
 
΅
Μ	variables
Νtrainable_variables
υlayer_metrics
φmetrics
χlayers
 ψlayer_regularization_losses
Ξregularization_losses
ωnon_trainable_variables
`^
VARIABLE_VALUEblock4/conv1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4/conv1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Π0
Ρ1

Π0
Ρ1
 
΅
?	variables
Σtrainable_variables
ϊlayer_metrics
ϋmetrics
όlayers
 ύlayer_regularization_losses
Τregularization_losses
ώnon_trainable_variables
 
 
 
΅
Φ	variables
Χtrainable_variables
?layer_metrics
metrics
layers
 layer_regularization_losses
Ψregularization_losses
non_trainable_variables
`^
VARIABLE_VALUEblock4/conv2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4/conv2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ϊ0
Ϋ1

Ϊ0
Ϋ1
 
΅
ά	variables
έtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
ήregularization_losses
non_trainable_variables
ge
VARIABLE_VALUEblock4/convshortcut/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEblock4/convshortcut/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

ΰ0
α1

ΰ0
α1
 
΅
β	variables
γtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
δregularization_losses
non_trainable_variables
 
 
 
΅
ζ	variables
ηtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
θregularization_losses
non_trainable_variables
 
 
 
΅
κ	variables
λtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
μregularization_losses
non_trainable_variables
 
 
 
΅
ξ	variables
οtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
πregularization_losses
non_trainable_variables
 
 
 
΅
ς	variables
σtrainable_variables
layer_metrics
metrics
layers
  layer_regularization_losses
τregularization_losses
‘non_trainable_variables
`^
VARIABLE_VALUEblock5/conv1/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5/conv1/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

φ0
χ1

φ0
χ1
 
΅
ψ	variables
ωtrainable_variables
’layer_metrics
£metrics
€layers
 ₯layer_regularization_losses
ϊregularization_losses
¦non_trainable_variables
 
 
 
΅
ό	variables
ύtrainable_variables
§layer_metrics
¨metrics
©layers
 ͺlayer_regularization_losses
ώregularization_losses
«non_trainable_variables
`^
VARIABLE_VALUEblock5/conv2/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5/conv2/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
¬layer_metrics
­metrics
?layers
 ―layer_regularization_losses
regularization_losses
°non_trainable_variables
ge
VARIABLE_VALUEblock5/convshortcut/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEblock5/convshortcut/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
±layer_metrics
²metrics
³layers
 ΄layer_regularization_losses
regularization_losses
΅non_trainable_variables
 
 
 
΅
	variables
trainable_variables
Άlayer_metrics
·metrics
Έlayers
 Ήlayer_regularization_losses
regularization_losses
Ίnon_trainable_variables
 
 
 
΅
	variables
trainable_variables
»layer_metrics
Όmetrics
½layers
 Ύlayer_regularization_losses
regularization_losses
Ώnon_trainable_variables
 
 
 
΅
	variables
trainable_variables
ΐlayer_metrics
Αmetrics
Βlayers
 Γlayer_regularization_losses
regularization_losses
Δnon_trainable_variables
 
 
 
΅
	variables
trainable_variables
Εlayer_metrics
Ζmetrics
Ηlayers
 Θlayer_regularization_losses
regularization_losses
Ιnon_trainable_variables
`^
VARIABLE_VALUEblock6/conv1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock6/conv1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
΅
	variables
trainable_variables
Κlayer_metrics
Λmetrics
Μlayers
 Νlayer_regularization_losses
 regularization_losses
Ξnon_trainable_variables
 
 
 
΅
’	variables
£trainable_variables
Οlayer_metrics
Πmetrics
Ρlayers
 ?layer_regularization_losses
€regularization_losses
Σnon_trainable_variables
`^
VARIABLE_VALUEblock6/conv2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock6/conv2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

¦0
§1

¦0
§1
 
΅
¨	variables
©trainable_variables
Τlayer_metrics
Υmetrics
Φlayers
 Χlayer_regularization_losses
ͺregularization_losses
Ψnon_trainable_variables
ge
VARIABLE_VALUEblock6/convshortcut/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEblock6/convshortcut/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

¬0
­1

¬0
­1
 
΅
?	variables
―trainable_variables
Ωlayer_metrics
Ϊmetrics
Ϋlayers
 άlayer_regularization_losses
°regularization_losses
έnon_trainable_variables
 
 
 
΅
²	variables
³trainable_variables
ήlayer_metrics
ίmetrics
ΰlayers
 αlayer_regularization_losses
΄regularization_losses
βnon_trainable_variables
 
 
 
΅
Ά	variables
·trainable_variables
γlayer_metrics
δmetrics
εlayers
 ζlayer_regularization_losses
Έregularization_losses
ηnon_trainable_variables
 
 
 
΅
Ί	variables
»trainable_variables
θlayer_metrics
ιmetrics
κlayers
 λlayer_regularization_losses
Όregularization_losses
μnon_trainable_variables
 
 
 
΅
Ύ	variables
Ώtrainable_variables
νlayer_metrics
ξmetrics
οlayers
 πlayer_regularization_losses
ΐregularization_losses
ρnon_trainable_variables
WU
VARIABLE_VALUE
fc1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

Β0
Γ1

Β0
Γ1
 
΅
Δ	variables
Εtrainable_variables
ςlayer_metrics
σmetrics
τlayers
 υlayer_regularization_losses
Ζregularization_losses
φnon_trainable_variables
WU
VARIABLE_VALUE
fc2/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

Θ0
Ι1

Θ0
Ι1
 
΅
Κ	variables
Λtrainable_variables
χlayer_metrics
ψmetrics
ωlayers
 ϊlayer_regularization_losses
Μregularization_losses
ϋnon_trainable_variables
WU
VARIABLE_VALUE
fc3/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc3/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

Ξ0
Ο1

Ξ0
Ο1
 
΅
Π	variables
Ρtrainable_variables
όlayer_metrics
ύmetrics
ώlayers
 ?layer_regularization_losses
?regularization_losses
non_trainable_variables
ZX
VARIABLE_VALUEfc_out/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEfc_out/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

Τ0
Υ1

Τ0
Υ1
 
΅
Φ	variables
Χtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
Ψregularization_losses
non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
Ξ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
|z
VARIABLE_VALUEAdam/conv01/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv01/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block0/conv1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block0/conv1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block0/conv2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block0/conv2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block1/conv1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block1/conv1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block1/conv2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block1/conv2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/conv1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block2/conv1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/conv2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block2/conv2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block2/convshortcut/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/convshortcut/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/conv1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block3/conv1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/conv2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block3/conv2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block3/convshortcut/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/convshortcut/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/conv1/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block4/conv1/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/conv2/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block4/conv2/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block4/convshortcut/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/convshortcut/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/conv1/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block5/conv1/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/conv2/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block5/conv2/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block5/convshortcut/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/convshortcut/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/conv1/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block6/conv1/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/conv2/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block6/conv2/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block6/convshortcut/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/convshortcut/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc1/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc1/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc2/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc2/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc3/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc3/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/fc_out/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc_out/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv01/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv01/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block0/conv1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block0/conv1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block0/conv2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block0/conv2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block1/conv1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block1/conv1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block1/conv2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block1/conv2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/conv1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block2/conv1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/conv2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block2/conv2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block2/convshortcut/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block2/convshortcut/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/conv1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block3/conv1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/conv2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/block3/conv2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block3/convshortcut/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block3/convshortcut/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/conv1/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block4/conv1/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/conv2/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block4/conv2/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block4/convshortcut/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block4/convshortcut/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/conv1/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block5/conv1/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/conv2/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block5/conv2/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block5/convshortcut/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block5/convshortcut/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/conv1/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block6/conv1/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/conv2/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/block6/conv2/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/block6/convshortcut/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/block6/convshortcut/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc1/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc1/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc2/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc2/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc3/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc3/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/fc_out/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc_out/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:?????????  H*
dtype0*$
shape:?????????  H
Ψ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv01/kernelconv01/biasblock0/conv1/kernelblock0/conv1/biasblock0/conv2/kernelblock0/conv2/biasblock1/conv1/kernelblock1/conv1/biasblock1/conv2/kernelblock1/conv2/biasblock2/conv1/kernelblock2/conv1/biasblock2/convshortcut/kernelblock2/convshortcut/biasblock2/conv2/kernelblock2/conv2/biasblock3/conv1/kernelblock3/conv1/biasblock3/convshortcut/kernelblock3/convshortcut/biasblock3/conv2/kernelblock3/conv2/biasblock4/conv1/kernelblock4/conv1/biasblock4/convshortcut/kernelblock4/convshortcut/biasblock4/conv2/kernelblock4/conv2/biasblock5/conv1/kernelblock5/conv1/biasblock5/convshortcut/kernelblock5/convshortcut/biasblock5/conv2/kernelblock5/conv2/biasblock6/conv1/kernelblock6/conv1/biasblock6/convshortcut/kernelblock6/convshortcut/biasblock6/conv2/kernelblock6/conv2/bias
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/biasfc_out/kernelfc_out/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_3746665
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ί7
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv01/kernel/Read/ReadVariableOpconv01/bias/Read/ReadVariableOp'block0/conv1/kernel/Read/ReadVariableOp%block0/conv1/bias/Read/ReadVariableOp'block0/conv2/kernel/Read/ReadVariableOp%block0/conv2/bias/Read/ReadVariableOp'block1/conv1/kernel/Read/ReadVariableOp%block1/conv1/bias/Read/ReadVariableOp'block1/conv2/kernel/Read/ReadVariableOp%block1/conv2/bias/Read/ReadVariableOp'block2/conv1/kernel/Read/ReadVariableOp%block2/conv1/bias/Read/ReadVariableOp'block2/conv2/kernel/Read/ReadVariableOp%block2/conv2/bias/Read/ReadVariableOp.block2/convshortcut/kernel/Read/ReadVariableOp,block2/convshortcut/bias/Read/ReadVariableOp'block3/conv1/kernel/Read/ReadVariableOp%block3/conv1/bias/Read/ReadVariableOp'block3/conv2/kernel/Read/ReadVariableOp%block3/conv2/bias/Read/ReadVariableOp.block3/convshortcut/kernel/Read/ReadVariableOp,block3/convshortcut/bias/Read/ReadVariableOp'block4/conv1/kernel/Read/ReadVariableOp%block4/conv1/bias/Read/ReadVariableOp'block4/conv2/kernel/Read/ReadVariableOp%block4/conv2/bias/Read/ReadVariableOp.block4/convshortcut/kernel/Read/ReadVariableOp,block4/convshortcut/bias/Read/ReadVariableOp'block5/conv1/kernel/Read/ReadVariableOp%block5/conv1/bias/Read/ReadVariableOp'block5/conv2/kernel/Read/ReadVariableOp%block5/conv2/bias/Read/ReadVariableOp.block5/convshortcut/kernel/Read/ReadVariableOp,block5/convshortcut/bias/Read/ReadVariableOp'block6/conv1/kernel/Read/ReadVariableOp%block6/conv1/bias/Read/ReadVariableOp'block6/conv2/kernel/Read/ReadVariableOp%block6/conv2/bias/Read/ReadVariableOp.block6/convshortcut/kernel/Read/ReadVariableOp,block6/convshortcut/bias/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpfc3/kernel/Read/ReadVariableOpfc3/bias/Read/ReadVariableOp!fc_out/kernel/Read/ReadVariableOpfc_out/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv01/kernel/m/Read/ReadVariableOp&Adam/conv01/bias/m/Read/ReadVariableOp.Adam/block0/conv1/kernel/m/Read/ReadVariableOp,Adam/block0/conv1/bias/m/Read/ReadVariableOp.Adam/block0/conv2/kernel/m/Read/ReadVariableOp,Adam/block0/conv2/bias/m/Read/ReadVariableOp.Adam/block1/conv1/kernel/m/Read/ReadVariableOp,Adam/block1/conv1/bias/m/Read/ReadVariableOp.Adam/block1/conv2/kernel/m/Read/ReadVariableOp,Adam/block1/conv2/bias/m/Read/ReadVariableOp.Adam/block2/conv1/kernel/m/Read/ReadVariableOp,Adam/block2/conv1/bias/m/Read/ReadVariableOp.Adam/block2/conv2/kernel/m/Read/ReadVariableOp,Adam/block2/conv2/bias/m/Read/ReadVariableOp5Adam/block2/convshortcut/kernel/m/Read/ReadVariableOp3Adam/block2/convshortcut/bias/m/Read/ReadVariableOp.Adam/block3/conv1/kernel/m/Read/ReadVariableOp,Adam/block3/conv1/bias/m/Read/ReadVariableOp.Adam/block3/conv2/kernel/m/Read/ReadVariableOp,Adam/block3/conv2/bias/m/Read/ReadVariableOp5Adam/block3/convshortcut/kernel/m/Read/ReadVariableOp3Adam/block3/convshortcut/bias/m/Read/ReadVariableOp.Adam/block4/conv1/kernel/m/Read/ReadVariableOp,Adam/block4/conv1/bias/m/Read/ReadVariableOp.Adam/block4/conv2/kernel/m/Read/ReadVariableOp,Adam/block4/conv2/bias/m/Read/ReadVariableOp5Adam/block4/convshortcut/kernel/m/Read/ReadVariableOp3Adam/block4/convshortcut/bias/m/Read/ReadVariableOp.Adam/block5/conv1/kernel/m/Read/ReadVariableOp,Adam/block5/conv1/bias/m/Read/ReadVariableOp.Adam/block5/conv2/kernel/m/Read/ReadVariableOp,Adam/block5/conv2/bias/m/Read/ReadVariableOp5Adam/block5/convshortcut/kernel/m/Read/ReadVariableOp3Adam/block5/convshortcut/bias/m/Read/ReadVariableOp.Adam/block6/conv1/kernel/m/Read/ReadVariableOp,Adam/block6/conv1/bias/m/Read/ReadVariableOp.Adam/block6/conv2/kernel/m/Read/ReadVariableOp,Adam/block6/conv2/bias/m/Read/ReadVariableOp5Adam/block6/convshortcut/kernel/m/Read/ReadVariableOp3Adam/block6/convshortcut/bias/m/Read/ReadVariableOp%Adam/fc1/kernel/m/Read/ReadVariableOp#Adam/fc1/bias/m/Read/ReadVariableOp%Adam/fc2/kernel/m/Read/ReadVariableOp#Adam/fc2/bias/m/Read/ReadVariableOp%Adam/fc3/kernel/m/Read/ReadVariableOp#Adam/fc3/bias/m/Read/ReadVariableOp(Adam/fc_out/kernel/m/Read/ReadVariableOp&Adam/fc_out/bias/m/Read/ReadVariableOp(Adam/conv01/kernel/v/Read/ReadVariableOp&Adam/conv01/bias/v/Read/ReadVariableOp.Adam/block0/conv1/kernel/v/Read/ReadVariableOp,Adam/block0/conv1/bias/v/Read/ReadVariableOp.Adam/block0/conv2/kernel/v/Read/ReadVariableOp,Adam/block0/conv2/bias/v/Read/ReadVariableOp.Adam/block1/conv1/kernel/v/Read/ReadVariableOp,Adam/block1/conv1/bias/v/Read/ReadVariableOp.Adam/block1/conv2/kernel/v/Read/ReadVariableOp,Adam/block1/conv2/bias/v/Read/ReadVariableOp.Adam/block2/conv1/kernel/v/Read/ReadVariableOp,Adam/block2/conv1/bias/v/Read/ReadVariableOp.Adam/block2/conv2/kernel/v/Read/ReadVariableOp,Adam/block2/conv2/bias/v/Read/ReadVariableOp5Adam/block2/convshortcut/kernel/v/Read/ReadVariableOp3Adam/block2/convshortcut/bias/v/Read/ReadVariableOp.Adam/block3/conv1/kernel/v/Read/ReadVariableOp,Adam/block3/conv1/bias/v/Read/ReadVariableOp.Adam/block3/conv2/kernel/v/Read/ReadVariableOp,Adam/block3/conv2/bias/v/Read/ReadVariableOp5Adam/block3/convshortcut/kernel/v/Read/ReadVariableOp3Adam/block3/convshortcut/bias/v/Read/ReadVariableOp.Adam/block4/conv1/kernel/v/Read/ReadVariableOp,Adam/block4/conv1/bias/v/Read/ReadVariableOp.Adam/block4/conv2/kernel/v/Read/ReadVariableOp,Adam/block4/conv2/bias/v/Read/ReadVariableOp5Adam/block4/convshortcut/kernel/v/Read/ReadVariableOp3Adam/block4/convshortcut/bias/v/Read/ReadVariableOp.Adam/block5/conv1/kernel/v/Read/ReadVariableOp,Adam/block5/conv1/bias/v/Read/ReadVariableOp.Adam/block5/conv2/kernel/v/Read/ReadVariableOp,Adam/block5/conv2/bias/v/Read/ReadVariableOp5Adam/block5/convshortcut/kernel/v/Read/ReadVariableOp3Adam/block5/convshortcut/bias/v/Read/ReadVariableOp.Adam/block6/conv1/kernel/v/Read/ReadVariableOp,Adam/block6/conv1/bias/v/Read/ReadVariableOp.Adam/block6/conv2/kernel/v/Read/ReadVariableOp,Adam/block6/conv2/bias/v/Read/ReadVariableOp5Adam/block6/convshortcut/kernel/v/Read/ReadVariableOp3Adam/block6/convshortcut/bias/v/Read/ReadVariableOp%Adam/fc1/kernel/v/Read/ReadVariableOp#Adam/fc1/bias/v/Read/ReadVariableOp%Adam/fc2/kernel/v/Read/ReadVariableOp#Adam/fc2/bias/v/Read/ReadVariableOp%Adam/fc3/kernel/v/Read/ReadVariableOp#Adam/fc3/bias/v/Read/ReadVariableOp(Adam/fc_out/kernel/v/Read/ReadVariableOp&Adam/fc_out/bias/v/Read/ReadVariableOpConst*©
Tin‘
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_3748775
ζ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv01/kernelconv01/biasblock0/conv1/kernelblock0/conv1/biasblock0/conv2/kernelblock0/conv2/biasblock1/conv1/kernelblock1/conv1/biasblock1/conv2/kernelblock1/conv2/biasblock2/conv1/kernelblock2/conv1/biasblock2/conv2/kernelblock2/conv2/biasblock2/convshortcut/kernelblock2/convshortcut/biasblock3/conv1/kernelblock3/conv1/biasblock3/conv2/kernelblock3/conv2/biasblock3/convshortcut/kernelblock3/convshortcut/biasblock4/conv1/kernelblock4/conv1/biasblock4/conv2/kernelblock4/conv2/biasblock4/convshortcut/kernelblock4/convshortcut/biasblock5/conv1/kernelblock5/conv1/biasblock5/conv2/kernelblock5/conv2/biasblock5/convshortcut/kernelblock5/convshortcut/biasblock6/conv1/kernelblock6/conv1/biasblock6/conv2/kernelblock6/conv2/biasblock6/convshortcut/kernelblock6/convshortcut/bias
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/biasfc_out/kernelfc_out/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv01/kernel/mAdam/conv01/bias/mAdam/block0/conv1/kernel/mAdam/block0/conv1/bias/mAdam/block0/conv2/kernel/mAdam/block0/conv2/bias/mAdam/block1/conv1/kernel/mAdam/block1/conv1/bias/mAdam/block1/conv2/kernel/mAdam/block1/conv2/bias/mAdam/block2/conv1/kernel/mAdam/block2/conv1/bias/mAdam/block2/conv2/kernel/mAdam/block2/conv2/bias/m!Adam/block2/convshortcut/kernel/mAdam/block2/convshortcut/bias/mAdam/block3/conv1/kernel/mAdam/block3/conv1/bias/mAdam/block3/conv2/kernel/mAdam/block3/conv2/bias/m!Adam/block3/convshortcut/kernel/mAdam/block3/convshortcut/bias/mAdam/block4/conv1/kernel/mAdam/block4/conv1/bias/mAdam/block4/conv2/kernel/mAdam/block4/conv2/bias/m!Adam/block4/convshortcut/kernel/mAdam/block4/convshortcut/bias/mAdam/block5/conv1/kernel/mAdam/block5/conv1/bias/mAdam/block5/conv2/kernel/mAdam/block5/conv2/bias/m!Adam/block5/convshortcut/kernel/mAdam/block5/convshortcut/bias/mAdam/block6/conv1/kernel/mAdam/block6/conv1/bias/mAdam/block6/conv2/kernel/mAdam/block6/conv2/bias/m!Adam/block6/convshortcut/kernel/mAdam/block6/convshortcut/bias/mAdam/fc1/kernel/mAdam/fc1/bias/mAdam/fc2/kernel/mAdam/fc2/bias/mAdam/fc3/kernel/mAdam/fc3/bias/mAdam/fc_out/kernel/mAdam/fc_out/bias/mAdam/conv01/kernel/vAdam/conv01/bias/vAdam/block0/conv1/kernel/vAdam/block0/conv1/bias/vAdam/block0/conv2/kernel/vAdam/block0/conv2/bias/vAdam/block1/conv1/kernel/vAdam/block1/conv1/bias/vAdam/block1/conv2/kernel/vAdam/block1/conv2/bias/vAdam/block2/conv1/kernel/vAdam/block2/conv1/bias/vAdam/block2/conv2/kernel/vAdam/block2/conv2/bias/v!Adam/block2/convshortcut/kernel/vAdam/block2/convshortcut/bias/vAdam/block3/conv1/kernel/vAdam/block3/conv1/bias/vAdam/block3/conv2/kernel/vAdam/block3/conv2/bias/v!Adam/block3/convshortcut/kernel/vAdam/block3/convshortcut/bias/vAdam/block4/conv1/kernel/vAdam/block4/conv1/bias/vAdam/block4/conv2/kernel/vAdam/block4/conv2/bias/v!Adam/block4/convshortcut/kernel/vAdam/block4/convshortcut/bias/vAdam/block5/conv1/kernel/vAdam/block5/conv1/bias/vAdam/block5/conv2/kernel/vAdam/block5/conv2/bias/v!Adam/block5/convshortcut/kernel/vAdam/block5/convshortcut/bias/vAdam/block6/conv1/kernel/vAdam/block6/conv1/bias/vAdam/block6/conv2/kernel/vAdam/block6/conv2/bias/v!Adam/block6/convshortcut/kernel/vAdam/block6/convshortcut/bias/vAdam/fc1/kernel/vAdam/fc1/bias/vAdam/fc2/kernel/vAdam/fc2/bias/vAdam/fc3/kernel/vAdam/fc3/bias/vAdam/fc_out/kernel/vAdam/fc_out/bias/v*¨
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_3749244Κξ 


.__inference_block6/conv2_layer_call_fn_3748072

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv2_layer_call_and_return_conditional_losses_37456182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block6/conv1_layer_call_and_return_conditional_losses_3748034

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_7_layer_call_and_return_conditional_losses_3747612

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_3_layer_call_and_return_conditional_losses_3747503

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Η
K
/__inference_activation_10_layer_call_fn_3747697

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_10_layer_call_and_return_conditional_losses_37451002
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_3747973

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_19_layer_call_fn_3747944

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_19_layer_call_and_return_conditional_losses_37454292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block5/conv2_layer_call_and_return_conditional_losses_3745473

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block1/conv1_layer_call_and_return_conditional_losses_3744867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3747481

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ε
J
.__inference_activation_7_layer_call_fn_3747617

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_37449942
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_15_layer_call_and_return_conditional_losses_3745284

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block0/conv2_layer_call_and_return_conditional_losses_3747467

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Η
K
/__inference_activation_21_layer_call_fn_3748002

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_21_layer_call_and_return_conditional_losses_37455072
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
l
@__inference_add_layer_call_and_return_conditional_losses_3747492
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????  2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :Z V
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
Η
K
/__inference_activation_13_layer_call_fn_3747784

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_13_layer_call_and_return_conditional_losses_37452172
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_16_layer_call_and_return_conditional_losses_3745349

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block3/conv1_layer_call_and_return_conditional_losses_3747707

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_21_layer_call_and_return_conditional_losses_3747997

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_3748082

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

«
@__inference_fc1_layer_call_and_return_conditional_losses_3748164

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block4/conv2_layer_call_and_return_conditional_losses_3747845

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_22_layer_call_and_return_conditional_losses_3748019

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ
l
B__inference_add_4_layer_call_and_return_conditional_losses_3745376

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3744821

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_10_layer_call_and_return_conditional_losses_3745100

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block4/conv1_layer_call_and_return_conditional_losses_3745263

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ρ
ή
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746196

inputs
conv01_3746041
conv01_3746043
block0_conv1_3746047
block0_conv1_3746049
block0_conv2_3746053
block0_conv2_3746055
block1_conv1_3746061
block1_conv1_3746063
block1_conv2_3746067
block1_conv2_3746069
block2_conv1_3746075
block2_conv1_3746077
block2_convshortcut_3746081
block2_convshortcut_3746083
block2_conv2_3746086
block2_conv2_3746088
block3_conv1_3746095
block3_conv1_3746097
block3_convshortcut_3746101
block3_convshortcut_3746103
block3_conv2_3746106
block3_conv2_3746108
block4_conv1_3746115
block4_conv1_3746117
block4_convshortcut_3746121
block4_convshortcut_3746123
block4_conv2_3746126
block4_conv2_3746128
block5_conv1_3746135
block5_conv1_3746137
block5_convshortcut_3746141
block5_convshortcut_3746143
block5_conv2_3746146
block5_conv2_3746148
block6_conv1_3746155
block6_conv1_3746157
block6_convshortcut_3746161
block6_convshortcut_3746163
block6_conv2_3746166
block6_conv2_3746168
fc1_3746175
fc1_3746177
fc2_3746180
fc2_3746182
fc3_3746185
fc3_3746187
fc_out_3746190
fc_out_3746192
identity’$block0/conv1/StatefulPartitionedCall’$block0/conv2/StatefulPartitionedCall’$block1/conv1/StatefulPartitionedCall’$block1/conv2/StatefulPartitionedCall’$block2/conv1/StatefulPartitionedCall’$block2/conv2/StatefulPartitionedCall’+block2/convshortcut/StatefulPartitionedCall’$block3/conv1/StatefulPartitionedCall’$block3/conv2/StatefulPartitionedCall’+block3/convshortcut/StatefulPartitionedCall’$block4/conv1/StatefulPartitionedCall’$block4/conv2/StatefulPartitionedCall’+block4/convshortcut/StatefulPartitionedCall’$block5/conv1/StatefulPartitionedCall’$block5/conv2/StatefulPartitionedCall’+block5/convshortcut/StatefulPartitionedCall’$block6/conv1/StatefulPartitionedCall’$block6/conv2/StatefulPartitionedCall’+block6/convshortcut/StatefulPartitionedCall’conv01/StatefulPartitionedCall’fc1/StatefulPartitionedCall’fc2/StatefulPartitionedCall’fc3/StatefulPartitionedCall’fc_out/StatefulPartitionedCall
conv01/StatefulPartitionedCallStatefulPartitionedCallinputsconv01_3746041conv01_3746043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv01_layer_call_and_return_conditional_losses_37447222 
conv01/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_37447432
activation/PartitionedCallΦ
$block0/conv1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0block0_conv1_3746047block0_conv1_3746049*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv1_layer_call_and_return_conditional_losses_37447612&
$block0/conv1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall-block0/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_37447822
activation_1/PartitionedCallΨ
$block0/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0block0_conv2_3746053block0_conv2_3746055*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv2_layer_call_and_return_conditional_losses_37448002&
$block0/conv2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall-block0/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_37448212
activation_2/PartitionedCall
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_37448352
add/PartitionedCall
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_37448492
activation_3/PartitionedCallΨ
$block1/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0block1_conv1_3746061block1_conv1_3746063*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv1_layer_call_and_return_conditional_losses_37448672&
$block1/conv1/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall-block1/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_37448882
activation_4/PartitionedCallΨ
$block1/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0block1_conv2_3746067block1_conv2_3746069*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv2_layer_call_and_return_conditional_losses_37449062&
$block1/conv2/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall-block1/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_37449272
activation_5/PartitionedCall‘
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_37449412
add_1/PartitionedCall
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_37449552
activation_6/PartitionedCallΨ
$block2/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_conv1_3746075block2_conv1_3746077*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv1_layer_call_and_return_conditional_losses_37449732&
$block2/conv1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall-block2/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_37449942
activation_7/PartitionedCallϋ
+block2/convshortcut/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_convshortcut_3746081block2_convshortcut_3746083*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_37450122-
+block2/convshortcut/StatefulPartitionedCallΨ
$block2/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0block2_conv2_3746086block2_conv2_3746088*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv2_layer_call_and_return_conditional_losses_37450382&
$block2/conv2/StatefulPartitionedCall
activation_8/PartitionedCallPartitionedCall-block2/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_8_layer_call_and_return_conditional_losses_37450592
activation_8/PartitionedCall
activation_9/PartitionedCallPartitionedCall4block2/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_9_layer_call_and_return_conditional_losses_37450722
activation_9/PartitionedCall‘
add_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_37450862
add_2/PartitionedCall
activation_10/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_10_layer_call_and_return_conditional_losses_37451002
activation_10/PartitionedCallΩ
$block3/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_conv1_3746095block3_conv1_3746097*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv1_layer_call_and_return_conditional_losses_37451182&
$block3/conv1/StatefulPartitionedCall
activation_11/PartitionedCallPartitionedCall-block3/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_11_layer_call_and_return_conditional_losses_37451392
activation_11/PartitionedCallό
+block3/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_convshortcut_3746101block3_convshortcut_3746103*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_37451572-
+block3/convshortcut/StatefulPartitionedCallΩ
$block3/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_11/PartitionedCall:output:0block3_conv2_3746106block3_conv2_3746108*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv2_layer_call_and_return_conditional_losses_37451832&
$block3/conv2/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall-block3/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_12_layer_call_and_return_conditional_losses_37452042
activation_12/PartitionedCall 
activation_13/PartitionedCallPartitionedCall4block3/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_13_layer_call_and_return_conditional_losses_37452172
activation_13/PartitionedCall£
add_3/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_37452312
add_3/PartitionedCall
activation_14/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_14_layer_call_and_return_conditional_losses_37452452
activation_14/PartitionedCallΩ
$block4/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_conv1_3746115block4_conv1_3746117*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv1_layer_call_and_return_conditional_losses_37452632&
$block4/conv1/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall-block4/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_15_layer_call_and_return_conditional_losses_37452842
activation_15/PartitionedCallό
+block4/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_convshortcut_3746121block4_convshortcut_3746123*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_37453022-
+block4/convshortcut/StatefulPartitionedCallΩ
$block4/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0block4_conv2_3746126block4_conv2_3746128*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv2_layer_call_and_return_conditional_losses_37453282&
$block4/conv2/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall-block4/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_16_layer_call_and_return_conditional_losses_37453492
activation_16/PartitionedCall 
activation_17/PartitionedCallPartitionedCall4block4/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_17_layer_call_and_return_conditional_losses_37453622
activation_17/PartitionedCall£
add_4/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_4_layer_call_and_return_conditional_losses_37453762
add_4/PartitionedCall
activation_18/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_18_layer_call_and_return_conditional_losses_37453902
activation_18/PartitionedCallΩ
$block5/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_conv1_3746135block5_conv1_3746137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv1_layer_call_and_return_conditional_losses_37454082&
$block5/conv1/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-block5/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_19_layer_call_and_return_conditional_losses_37454292
activation_19/PartitionedCallό
+block5/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_convshortcut_3746141block5_convshortcut_3746143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_37454472-
+block5/convshortcut/StatefulPartitionedCallΩ
$block5/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0block5_conv2_3746146block5_conv2_3746148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv2_layer_call_and_return_conditional_losses_37454732&
$block5/conv2/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall-block5/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_20_layer_call_and_return_conditional_losses_37454942
activation_20/PartitionedCall 
activation_21/PartitionedCallPartitionedCall4block5/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_21_layer_call_and_return_conditional_losses_37455072
activation_21/PartitionedCall£
add_5/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_5_layer_call_and_return_conditional_losses_37455212
add_5/PartitionedCall
activation_22/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_22_layer_call_and_return_conditional_losses_37455352
activation_22/PartitionedCallΩ
$block6/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_conv1_3746155block6_conv1_3746157*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv1_layer_call_and_return_conditional_losses_37455532&
$block6/conv1/StatefulPartitionedCall
activation_23/PartitionedCallPartitionedCall-block6/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_23_layer_call_and_return_conditional_losses_37455742
activation_23/PartitionedCallό
+block6/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_convshortcut_3746161block6_convshortcut_3746163*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_37455922-
+block6/convshortcut/StatefulPartitionedCallΩ
$block6/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0block6_conv2_3746166block6_conv2_3746168*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv2_layer_call_and_return_conditional_losses_37456182&
$block6/conv2/StatefulPartitionedCall
activation_24/PartitionedCallPartitionedCall-block6/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_24_layer_call_and_return_conditional_losses_37456392
activation_24/PartitionedCall 
activation_25/PartitionedCallPartitionedCall4block6/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_25_layer_call_and_return_conditional_losses_37456522
activation_25/PartitionedCall£
add_6/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_6_layer_call_and_return_conditional_losses_37456662
add_6/PartitionedCall
activation_26/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_26_layer_call_and_return_conditional_losses_37456802
activation_26/PartitionedCall¬
fc1/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0fc1_3746175fc1_3746177*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_37457192
fc1/StatefulPartitionedCallͺ
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0fc2_3746180fc2_3746182*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_37457662
fc2/StatefulPartitionedCallͺ
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0fc3_3746185fc3_3746187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc3_layer_call_and_return_conditional_losses_37458132
fc3/StatefulPartitionedCallΈ
fc_out/StatefulPartitionedCallStatefulPartitionedCall$fc3/StatefulPartitionedCall:output:0fc_out_3746190fc_out_3746192*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_fc_out_layer_call_and_return_conditional_losses_37458602 
fc_out/StatefulPartitionedCall§
IdentityIdentity'fc_out/StatefulPartitionedCall:output:0%^block0/conv1/StatefulPartitionedCall%^block0/conv2/StatefulPartitionedCall%^block1/conv1/StatefulPartitionedCall%^block1/conv2/StatefulPartitionedCall%^block2/conv1/StatefulPartitionedCall%^block2/conv2/StatefulPartitionedCall,^block2/convshortcut/StatefulPartitionedCall%^block3/conv1/StatefulPartitionedCall%^block3/conv2/StatefulPartitionedCall,^block3/convshortcut/StatefulPartitionedCall%^block4/conv1/StatefulPartitionedCall%^block4/conv2/StatefulPartitionedCall,^block4/convshortcut/StatefulPartitionedCall%^block5/conv1/StatefulPartitionedCall%^block5/conv2/StatefulPartitionedCall,^block5/convshortcut/StatefulPartitionedCall%^block6/conv1/StatefulPartitionedCall%^block6/conv2/StatefulPartitionedCall,^block6/convshortcut/StatefulPartitionedCall^conv01/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall^fc_out/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::2L
$block0/conv1/StatefulPartitionedCall$block0/conv1/StatefulPartitionedCall2L
$block0/conv2/StatefulPartitionedCall$block0/conv2/StatefulPartitionedCall2L
$block1/conv1/StatefulPartitionedCall$block1/conv1/StatefulPartitionedCall2L
$block1/conv2/StatefulPartitionedCall$block1/conv2/StatefulPartitionedCall2L
$block2/conv1/StatefulPartitionedCall$block2/conv1/StatefulPartitionedCall2L
$block2/conv2/StatefulPartitionedCall$block2/conv2/StatefulPartitionedCall2Z
+block2/convshortcut/StatefulPartitionedCall+block2/convshortcut/StatefulPartitionedCall2L
$block3/conv1/StatefulPartitionedCall$block3/conv1/StatefulPartitionedCall2L
$block3/conv2/StatefulPartitionedCall$block3/conv2/StatefulPartitionedCall2Z
+block3/convshortcut/StatefulPartitionedCall+block3/convshortcut/StatefulPartitionedCall2L
$block4/conv1/StatefulPartitionedCall$block4/conv1/StatefulPartitionedCall2L
$block4/conv2/StatefulPartitionedCall$block4/conv2/StatefulPartitionedCall2Z
+block4/convshortcut/StatefulPartitionedCall+block4/convshortcut/StatefulPartitionedCall2L
$block5/conv1/StatefulPartitionedCall$block5/conv1/StatefulPartitionedCall2L
$block5/conv2/StatefulPartitionedCall$block5/conv2/StatefulPartitionedCall2Z
+block5/convshortcut/StatefulPartitionedCall+block5/convshortcut/StatefulPartitionedCall2L
$block6/conv1/StatefulPartitionedCall$block6/conv1/StatefulPartitionedCall2L
$block6/conv2/StatefulPartitionedCall$block6/conv2/StatefulPartitionedCall2Z
+block6/convshortcut/StatefulPartitionedCall+block6/convshortcut/StatefulPartitionedCall2@
conv01/StatefulPartitionedCallconv01/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall2@
fc_out/StatefulPartitionedCallfc_out/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Ω
c
G__inference_activation_layer_call_and_return_conditional_losses_3744743

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block2/conv1_layer_call_and_return_conditional_losses_3747598

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_7_layer_call_and_return_conditional_losses_3744994

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_8_layer_call_and_return_conditional_losses_3745059

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block4/conv1_layer_call_and_return_conditional_losses_3747816

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_21_layer_call_and_return_conditional_losses_3745507

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_17_layer_call_and_return_conditional_losses_3747888

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_26_layer_call_and_return_conditional_losses_3745680

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_4_layer_call_and_return_conditional_losses_3744888

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block0/conv2_layer_call_and_return_conditional_losses_3744800

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs


.__inference_block2/conv2_layer_call_fn_3747636

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv2_layer_call_and_return_conditional_losses_37450382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
n
B__inference_add_1_layer_call_and_return_conditional_losses_3747572
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????  2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :Z V
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
Ϋ
e
I__inference_activation_9_layer_call_and_return_conditional_losses_3747670

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

«
@__inference_fc2_layer_call_and_return_conditional_losses_3748204

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block0/conv1_layer_call_and_return_conditional_losses_3744761

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block3/conv2_layer_call_and_return_conditional_losses_3747736

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_14_layer_call_fn_3747806

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_14_layer_call_and_return_conditional_losses_37452452
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


5__inference_block6/convshortcut_layer_call_fn_3748091

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_37455922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block0/conv1_layer_call_fn_3747447

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv1_layer_call_and_return_conditional_losses_37447612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs


.__inference_block1/conv2_layer_call_fn_3747556

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv2_layer_call_and_return_conditional_losses_37449062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
³ͺ
‘B
 __inference__traced_save_3748775
file_prefix,
(savev2_conv01_kernel_read_readvariableop*
&savev2_conv01_bias_read_readvariableop2
.savev2_block0_conv1_kernel_read_readvariableop0
,savev2_block0_conv1_bias_read_readvariableop2
.savev2_block0_conv2_kernel_read_readvariableop0
,savev2_block0_conv2_bias_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop9
5savev2_block2_convshortcut_kernel_read_readvariableop7
3savev2_block2_convshortcut_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop9
5savev2_block3_convshortcut_kernel_read_readvariableop7
3savev2_block3_convshortcut_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop9
5savev2_block4_convshortcut_kernel_read_readvariableop7
3savev2_block4_convshortcut_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop9
5savev2_block5_convshortcut_kernel_read_readvariableop7
3savev2_block5_convshortcut_bias_read_readvariableop2
.savev2_block6_conv1_kernel_read_readvariableop0
,savev2_block6_conv1_bias_read_readvariableop2
.savev2_block6_conv2_kernel_read_readvariableop0
,savev2_block6_conv2_bias_read_readvariableop9
5savev2_block6_convshortcut_kernel_read_readvariableop7
3savev2_block6_convshortcut_bias_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop)
%savev2_fc2_kernel_read_readvariableop'
#savev2_fc2_bias_read_readvariableop)
%savev2_fc3_kernel_read_readvariableop'
#savev2_fc3_bias_read_readvariableop,
(savev2_fc_out_kernel_read_readvariableop*
&savev2_fc_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv01_kernel_m_read_readvariableop1
-savev2_adam_conv01_bias_m_read_readvariableop9
5savev2_adam_block0_conv1_kernel_m_read_readvariableop7
3savev2_adam_block0_conv1_bias_m_read_readvariableop9
5savev2_adam_block0_conv2_kernel_m_read_readvariableop7
3savev2_adam_block0_conv2_bias_m_read_readvariableop9
5savev2_adam_block1_conv1_kernel_m_read_readvariableop7
3savev2_adam_block1_conv1_bias_m_read_readvariableop9
5savev2_adam_block1_conv2_kernel_m_read_readvariableop7
3savev2_adam_block1_conv2_bias_m_read_readvariableop9
5savev2_adam_block2_conv1_kernel_m_read_readvariableop7
3savev2_adam_block2_conv1_bias_m_read_readvariableop9
5savev2_adam_block2_conv2_kernel_m_read_readvariableop7
3savev2_adam_block2_conv2_bias_m_read_readvariableop@
<savev2_adam_block2_convshortcut_kernel_m_read_readvariableop>
:savev2_adam_block2_convshortcut_bias_m_read_readvariableop9
5savev2_adam_block3_conv1_kernel_m_read_readvariableop7
3savev2_adam_block3_conv1_bias_m_read_readvariableop9
5savev2_adam_block3_conv2_kernel_m_read_readvariableop7
3savev2_adam_block3_conv2_bias_m_read_readvariableop@
<savev2_adam_block3_convshortcut_kernel_m_read_readvariableop>
:savev2_adam_block3_convshortcut_bias_m_read_readvariableop9
5savev2_adam_block4_conv1_kernel_m_read_readvariableop7
3savev2_adam_block4_conv1_bias_m_read_readvariableop9
5savev2_adam_block4_conv2_kernel_m_read_readvariableop7
3savev2_adam_block4_conv2_bias_m_read_readvariableop@
<savev2_adam_block4_convshortcut_kernel_m_read_readvariableop>
:savev2_adam_block4_convshortcut_bias_m_read_readvariableop9
5savev2_adam_block5_conv1_kernel_m_read_readvariableop7
3savev2_adam_block5_conv1_bias_m_read_readvariableop9
5savev2_adam_block5_conv2_kernel_m_read_readvariableop7
3savev2_adam_block5_conv2_bias_m_read_readvariableop@
<savev2_adam_block5_convshortcut_kernel_m_read_readvariableop>
:savev2_adam_block5_convshortcut_bias_m_read_readvariableop9
5savev2_adam_block6_conv1_kernel_m_read_readvariableop7
3savev2_adam_block6_conv1_bias_m_read_readvariableop9
5savev2_adam_block6_conv2_kernel_m_read_readvariableop7
3savev2_adam_block6_conv2_bias_m_read_readvariableop@
<savev2_adam_block6_convshortcut_kernel_m_read_readvariableop>
:savev2_adam_block6_convshortcut_bias_m_read_readvariableop0
,savev2_adam_fc1_kernel_m_read_readvariableop.
*savev2_adam_fc1_bias_m_read_readvariableop0
,savev2_adam_fc2_kernel_m_read_readvariableop.
*savev2_adam_fc2_bias_m_read_readvariableop0
,savev2_adam_fc3_kernel_m_read_readvariableop.
*savev2_adam_fc3_bias_m_read_readvariableop3
/savev2_adam_fc_out_kernel_m_read_readvariableop1
-savev2_adam_fc_out_bias_m_read_readvariableop3
/savev2_adam_conv01_kernel_v_read_readvariableop1
-savev2_adam_conv01_bias_v_read_readvariableop9
5savev2_adam_block0_conv1_kernel_v_read_readvariableop7
3savev2_adam_block0_conv1_bias_v_read_readvariableop9
5savev2_adam_block0_conv2_kernel_v_read_readvariableop7
3savev2_adam_block0_conv2_bias_v_read_readvariableop9
5savev2_adam_block1_conv1_kernel_v_read_readvariableop7
3savev2_adam_block1_conv1_bias_v_read_readvariableop9
5savev2_adam_block1_conv2_kernel_v_read_readvariableop7
3savev2_adam_block1_conv2_bias_v_read_readvariableop9
5savev2_adam_block2_conv1_kernel_v_read_readvariableop7
3savev2_adam_block2_conv1_bias_v_read_readvariableop9
5savev2_adam_block2_conv2_kernel_v_read_readvariableop7
3savev2_adam_block2_conv2_bias_v_read_readvariableop@
<savev2_adam_block2_convshortcut_kernel_v_read_readvariableop>
:savev2_adam_block2_convshortcut_bias_v_read_readvariableop9
5savev2_adam_block3_conv1_kernel_v_read_readvariableop7
3savev2_adam_block3_conv1_bias_v_read_readvariableop9
5savev2_adam_block3_conv2_kernel_v_read_readvariableop7
3savev2_adam_block3_conv2_bias_v_read_readvariableop@
<savev2_adam_block3_convshortcut_kernel_v_read_readvariableop>
:savev2_adam_block3_convshortcut_bias_v_read_readvariableop9
5savev2_adam_block4_conv1_kernel_v_read_readvariableop7
3savev2_adam_block4_conv1_bias_v_read_readvariableop9
5savev2_adam_block4_conv2_kernel_v_read_readvariableop7
3savev2_adam_block4_conv2_bias_v_read_readvariableop@
<savev2_adam_block4_convshortcut_kernel_v_read_readvariableop>
:savev2_adam_block4_convshortcut_bias_v_read_readvariableop9
5savev2_adam_block5_conv1_kernel_v_read_readvariableop7
3savev2_adam_block5_conv1_bias_v_read_readvariableop9
5savev2_adam_block5_conv2_kernel_v_read_readvariableop7
3savev2_adam_block5_conv2_bias_v_read_readvariableop@
<savev2_adam_block5_convshortcut_kernel_v_read_readvariableop>
:savev2_adam_block5_convshortcut_bias_v_read_readvariableop9
5savev2_adam_block6_conv1_kernel_v_read_readvariableop7
3savev2_adam_block6_conv1_bias_v_read_readvariableop9
5savev2_adam_block6_conv2_kernel_v_read_readvariableop7
3savev2_adam_block6_conv2_bias_v_read_readvariableop@
<savev2_adam_block6_convshortcut_kernel_v_read_readvariableop>
:savev2_adam_block6_convshortcut_bias_v_read_readvariableop0
,savev2_adam_fc1_kernel_v_read_readvariableop.
*savev2_adam_fc1_bias_v_read_readvariableop0
,savev2_adam_fc2_kernel_v_read_readvariableop.
*savev2_adam_fc2_bias_v_read_readvariableop0
,savev2_adam_fc3_kernel_v_read_readvariableop.
*savev2_adam_fc3_bias_v_read_readvariableop3
/savev2_adam_fc_out_kernel_v_read_readvariableop1
-savev2_adam_fc_out_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72de89c8c97d4b068dc843dfc0deb40a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename€X
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*΅W
value«WB¨WB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesΑ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Κ
valueΐB½B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices§?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv01_kernel_read_readvariableop&savev2_conv01_bias_read_readvariableop.savev2_block0_conv1_kernel_read_readvariableop,savev2_block0_conv1_bias_read_readvariableop.savev2_block0_conv2_kernel_read_readvariableop,savev2_block0_conv2_bias_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop5savev2_block2_convshortcut_kernel_read_readvariableop3savev2_block2_convshortcut_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop5savev2_block3_convshortcut_kernel_read_readvariableop3savev2_block3_convshortcut_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop5savev2_block4_convshortcut_kernel_read_readvariableop3savev2_block4_convshortcut_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop5savev2_block5_convshortcut_kernel_read_readvariableop3savev2_block5_convshortcut_bias_read_readvariableop.savev2_block6_conv1_kernel_read_readvariableop,savev2_block6_conv1_bias_read_readvariableop.savev2_block6_conv2_kernel_read_readvariableop,savev2_block6_conv2_bias_read_readvariableop5savev2_block6_convshortcut_kernel_read_readvariableop3savev2_block6_convshortcut_bias_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop%savev2_fc3_kernel_read_readvariableop#savev2_fc3_bias_read_readvariableop(savev2_fc_out_kernel_read_readvariableop&savev2_fc_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv01_kernel_m_read_readvariableop-savev2_adam_conv01_bias_m_read_readvariableop5savev2_adam_block0_conv1_kernel_m_read_readvariableop3savev2_adam_block0_conv1_bias_m_read_readvariableop5savev2_adam_block0_conv2_kernel_m_read_readvariableop3savev2_adam_block0_conv2_bias_m_read_readvariableop5savev2_adam_block1_conv1_kernel_m_read_readvariableop3savev2_adam_block1_conv1_bias_m_read_readvariableop5savev2_adam_block1_conv2_kernel_m_read_readvariableop3savev2_adam_block1_conv2_bias_m_read_readvariableop5savev2_adam_block2_conv1_kernel_m_read_readvariableop3savev2_adam_block2_conv1_bias_m_read_readvariableop5savev2_adam_block2_conv2_kernel_m_read_readvariableop3savev2_adam_block2_conv2_bias_m_read_readvariableop<savev2_adam_block2_convshortcut_kernel_m_read_readvariableop:savev2_adam_block2_convshortcut_bias_m_read_readvariableop5savev2_adam_block3_conv1_kernel_m_read_readvariableop3savev2_adam_block3_conv1_bias_m_read_readvariableop5savev2_adam_block3_conv2_kernel_m_read_readvariableop3savev2_adam_block3_conv2_bias_m_read_readvariableop<savev2_adam_block3_convshortcut_kernel_m_read_readvariableop:savev2_adam_block3_convshortcut_bias_m_read_readvariableop5savev2_adam_block4_conv1_kernel_m_read_readvariableop3savev2_adam_block4_conv1_bias_m_read_readvariableop5savev2_adam_block4_conv2_kernel_m_read_readvariableop3savev2_adam_block4_conv2_bias_m_read_readvariableop<savev2_adam_block4_convshortcut_kernel_m_read_readvariableop:savev2_adam_block4_convshortcut_bias_m_read_readvariableop5savev2_adam_block5_conv1_kernel_m_read_readvariableop3savev2_adam_block5_conv1_bias_m_read_readvariableop5savev2_adam_block5_conv2_kernel_m_read_readvariableop3savev2_adam_block5_conv2_bias_m_read_readvariableop<savev2_adam_block5_convshortcut_kernel_m_read_readvariableop:savev2_adam_block5_convshortcut_bias_m_read_readvariableop5savev2_adam_block6_conv1_kernel_m_read_readvariableop3savev2_adam_block6_conv1_bias_m_read_readvariableop5savev2_adam_block6_conv2_kernel_m_read_readvariableop3savev2_adam_block6_conv2_bias_m_read_readvariableop<savev2_adam_block6_convshortcut_kernel_m_read_readvariableop:savev2_adam_block6_convshortcut_bias_m_read_readvariableop,savev2_adam_fc1_kernel_m_read_readvariableop*savev2_adam_fc1_bias_m_read_readvariableop,savev2_adam_fc2_kernel_m_read_readvariableop*savev2_adam_fc2_bias_m_read_readvariableop,savev2_adam_fc3_kernel_m_read_readvariableop*savev2_adam_fc3_bias_m_read_readvariableop/savev2_adam_fc_out_kernel_m_read_readvariableop-savev2_adam_fc_out_bias_m_read_readvariableop/savev2_adam_conv01_kernel_v_read_readvariableop-savev2_adam_conv01_bias_v_read_readvariableop5savev2_adam_block0_conv1_kernel_v_read_readvariableop3savev2_adam_block0_conv1_bias_v_read_readvariableop5savev2_adam_block0_conv2_kernel_v_read_readvariableop3savev2_adam_block0_conv2_bias_v_read_readvariableop5savev2_adam_block1_conv1_kernel_v_read_readvariableop3savev2_adam_block1_conv1_bias_v_read_readvariableop5savev2_adam_block1_conv2_kernel_v_read_readvariableop3savev2_adam_block1_conv2_bias_v_read_readvariableop5savev2_adam_block2_conv1_kernel_v_read_readvariableop3savev2_adam_block2_conv1_bias_v_read_readvariableop5savev2_adam_block2_conv2_kernel_v_read_readvariableop3savev2_adam_block2_conv2_bias_v_read_readvariableop<savev2_adam_block2_convshortcut_kernel_v_read_readvariableop:savev2_adam_block2_convshortcut_bias_v_read_readvariableop5savev2_adam_block3_conv1_kernel_v_read_readvariableop3savev2_adam_block3_conv1_bias_v_read_readvariableop5savev2_adam_block3_conv2_kernel_v_read_readvariableop3savev2_adam_block3_conv2_bias_v_read_readvariableop<savev2_adam_block3_convshortcut_kernel_v_read_readvariableop:savev2_adam_block3_convshortcut_bias_v_read_readvariableop5savev2_adam_block4_conv1_kernel_v_read_readvariableop3savev2_adam_block4_conv1_bias_v_read_readvariableop5savev2_adam_block4_conv2_kernel_v_read_readvariableop3savev2_adam_block4_conv2_bias_v_read_readvariableop<savev2_adam_block4_convshortcut_kernel_v_read_readvariableop:savev2_adam_block4_convshortcut_bias_v_read_readvariableop5savev2_adam_block5_conv1_kernel_v_read_readvariableop3savev2_adam_block5_conv1_bias_v_read_readvariableop5savev2_adam_block5_conv2_kernel_v_read_readvariableop3savev2_adam_block5_conv2_bias_v_read_readvariableop<savev2_adam_block5_convshortcut_kernel_v_read_readvariableop:savev2_adam_block5_convshortcut_bias_v_read_readvariableop5savev2_adam_block6_conv1_kernel_v_read_readvariableop3savev2_adam_block6_conv1_bias_v_read_readvariableop5savev2_adam_block6_conv2_kernel_v_read_readvariableop3savev2_adam_block6_conv2_bias_v_read_readvariableop<savev2_adam_block6_convshortcut_kernel_v_read_readvariableop:savev2_adam_block6_convshortcut_bias_v_read_readvariableop,savev2_adam_fc1_kernel_v_read_readvariableop*savev2_adam_fc1_bias_v_read_readvariableop,savev2_adam_fc2_kernel_v_read_readvariableop*savev2_adam_fc2_bias_v_read_readvariableop,savev2_adam_fc3_kernel_v_read_readvariableop*savev2_adam_fc3_bias_v_read_readvariableop/savev2_adam_fc_out_kernel_v_read_readvariableop-savev2_adam_fc_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *«
dtypes 
2	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ϊ
_input_shapesΘ
Ε: :H::::::::::::::::::::::::::::::::::::::::
::
::
::	:: : : : : : : : : :H::::::::::::::::::::::::::::::::::::::::
::
::
::	::H::::::::::::::::::::::::::::::::::::::::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:H:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::.#*
(
_output_shapes
::!$

_output_shapes	
::.%*
(
_output_shapes
::!&

_output_shapes	
::.'*
(
_output_shapes
::!(

_output_shapes	
::&)"
 
_output_shapes
:
:!*

_output_shapes	
::&+"
 
_output_shapes
:
:!,

_output_shapes	
::&-"
 
_output_shapes
:
:!.

_output_shapes	
::%/!

_output_shapes
:	: 0

_output_shapes
::1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :-:)
'
_output_shapes
:H:!;

_output_shapes	
::.<*
(
_output_shapes
::!=

_output_shapes	
::.>*
(
_output_shapes
::!?

_output_shapes	
::.@*
(
_output_shapes
::!A

_output_shapes	
::.B*
(
_output_shapes
::!C

_output_shapes	
::.D*
(
_output_shapes
::!E

_output_shapes	
::.F*
(
_output_shapes
::!G

_output_shapes	
::.H*
(
_output_shapes
::!I

_output_shapes	
::.J*
(
_output_shapes
::!K

_output_shapes	
::.L*
(
_output_shapes
::!M

_output_shapes	
::.N*
(
_output_shapes
::!O

_output_shapes	
::.P*
(
_output_shapes
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::.T*
(
_output_shapes
::!U

_output_shapes	
::.V*
(
_output_shapes
::!W

_output_shapes	
::.X*
(
_output_shapes
::!Y

_output_shapes	
::.Z*
(
_output_shapes
::![

_output_shapes	
::.\*
(
_output_shapes
::!]

_output_shapes	
::.^*
(
_output_shapes
::!_

_output_shapes	
::.`*
(
_output_shapes
::!a

_output_shapes	
::&b"
 
_output_shapes
:
:!c

_output_shapes	
::&d"
 
_output_shapes
:
:!e

_output_shapes	
::&f"
 
_output_shapes
:
:!g

_output_shapes	
::%h!

_output_shapes
:	: i

_output_shapes
::-j)
'
_output_shapes
:H:!k

_output_shapes	
::.l*
(
_output_shapes
::!m

_output_shapes	
::.n*
(
_output_shapes
::!o

_output_shapes	
::.p*
(
_output_shapes
::!q

_output_shapes	
::.r*
(
_output_shapes
::!s

_output_shapes	
::.t*
(
_output_shapes
::!u

_output_shapes	
::.v*
(
_output_shapes
::!w

_output_shapes	
::.x*
(
_output_shapes
::!y

_output_shapes	
::.z*
(
_output_shapes
::!{

_output_shapes	
::.|*
(
_output_shapes
::!}

_output_shapes	
::.~*
(
_output_shapes
::!

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::&!

_output_shapes
:	:!

_output_shapes
::

_output_shapes
: 
²
±
I__inference_block2/conv1_layer_call_and_return_conditional_losses_3744973

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block6/conv2_layer_call_and_return_conditional_losses_3748063

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_24_layer_call_fn_3748101

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_24_layer_call_and_return_conditional_losses_37456392
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_13_layer_call_and_return_conditional_losses_3745217

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Α
H
,__inference_activation_layer_call_fn_3747428

inputs
identityΣ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_37447432
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block1/conv2_layer_call_and_return_conditional_losses_3744906

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs


.__inference_block4/conv1_layer_call_fn_3747825

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv1_layer_call_and_return_conditional_losses_37452632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
n
B__inference_add_2_layer_call_and_return_conditional_losses_3747681
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_18_layer_call_and_return_conditional_losses_3745390

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_26_layer_call_fn_3748133

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_26_layer_call_and_return_conditional_losses_37456802
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ
l
B__inference_add_2_layer_call_and_return_conditional_losses_3745086

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
J
.__inference_activation_4_layer_call_fn_3747537

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_37448882
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_8_layer_call_and_return_conditional_losses_3747660

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

Τ
(__inference_MTFNet_layer_call_fn_3747298

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_MTFNet_layer_call_and_return_conditional_losses_37461962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
·
?
C__inference_MTFNet_layer_call_and_return_conditional_losses_3747197

inputs)
%conv01_conv2d_readvariableop_resource*
&conv01_biasadd_readvariableop_resource/
+block0_conv1_conv2d_readvariableop_resource0
,block0_conv1_biasadd_readvariableop_resource/
+block0_conv2_conv2d_readvariableop_resource0
,block0_conv2_biasadd_readvariableop_resource/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource6
2block2_convshortcut_conv2d_readvariableop_resource7
3block2_convshortcut_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource6
2block3_convshortcut_conv2d_readvariableop_resource7
3block3_convshortcut_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource6
2block4_convshortcut_conv2d_readvariableop_resource7
3block4_convshortcut_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource6
2block5_convshortcut_conv2d_readvariableop_resource7
3block5_convshortcut_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block6_conv1_conv2d_readvariableop_resource0
,block6_conv1_biasadd_readvariableop_resource6
2block6_convshortcut_conv2d_readvariableop_resource7
3block6_convshortcut_biasadd_readvariableop_resource/
+block6_conv2_conv2d_readvariableop_resource0
,block6_conv2_biasadd_readvariableop_resource)
%fc1_tensordot_readvariableop_resource'
#fc1_biasadd_readvariableop_resource)
%fc2_tensordot_readvariableop_resource'
#fc2_biasadd_readvariableop_resource)
%fc3_tensordot_readvariableop_resource'
#fc3_biasadd_readvariableop_resource,
(fc_out_tensordot_readvariableop_resource*
&fc_out_biasadd_readvariableop_resource
identity«
conv01/Conv2D/ReadVariableOpReadVariableOp%conv01_conv2d_readvariableop_resource*'
_output_shapes
:H*
dtype02
conv01/Conv2D/ReadVariableOpΉ
conv01/Conv2DConv2Dinputs$conv01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv01/Conv2D’
conv01/BiasAdd/ReadVariableOpReadVariableOp&conv01_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv01/BiasAdd/ReadVariableOp₯
conv01/BiasAddBiasAddconv01/Conv2D:output:0%conv01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
conv01/BiasAdd~
activation/ReluReluconv01/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation/ReluΎ
"block0/conv1/Conv2D/ReadVariableOpReadVariableOp+block0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block0/conv1/Conv2D/ReadVariableOpβ
block0/conv1/Conv2DConv2Dactivation/Relu:activations:0*block0/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block0/conv1/Conv2D΄
#block0/conv1/BiasAdd/ReadVariableOpReadVariableOp,block0_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block0/conv1/BiasAdd/ReadVariableOp½
block0/conv1/BiasAddBiasAddblock0/conv1/Conv2D:output:0+block0/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block0/conv1/BiasAdd
activation_1/ReluRelublock0/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_1/ReluΎ
"block0/conv2/Conv2D/ReadVariableOpReadVariableOp+block0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block0/conv2/Conv2D/ReadVariableOpδ
block0/conv2/Conv2DConv2Dactivation_1/Relu:activations:0*block0/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block0/conv2/Conv2D΄
#block0/conv2/BiasAdd/ReadVariableOpReadVariableOp,block0_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block0/conv2/BiasAdd/ReadVariableOp½
block0/conv2/BiasAddBiasAddblock0/conv2/Conv2D:output:0+block0/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block0/conv2/BiasAdd
activation_2/ReluRelublock0/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_2/Relu
add/addAddV2activation_2/Relu:activations:0activation/Relu:activations:0*
T0*0
_output_shapes
:?????????  2	
add/addv
activation_3/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????  2
activation_3/ReluΎ
"block1/conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block1/conv1/Conv2D/ReadVariableOpδ
block1/conv1/Conv2DConv2Dactivation_3/Relu:activations:0*block1/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block1/conv1/Conv2D΄
#block1/conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block1/conv1/BiasAdd/ReadVariableOp½
block1/conv1/BiasAddBiasAddblock1/conv1/Conv2D:output:0+block1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block1/conv1/BiasAdd
activation_4/ReluRelublock1/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_4/ReluΎ
"block1/conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block1/conv2/Conv2D/ReadVariableOpδ
block1/conv2/Conv2DConv2Dactivation_4/Relu:activations:0*block1/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block1/conv2/Conv2D΄
#block1/conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block1/conv2/BiasAdd/ReadVariableOp½
block1/conv2/BiasAddBiasAddblock1/conv2/Conv2D:output:0+block1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block1/conv2/BiasAdd
activation_5/ReluRelublock1/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_5/Relu
	add_1/addAddV2activation_5/Relu:activations:0activation_3/Relu:activations:0*
T0*0
_output_shapes
:?????????  2
	add_1/addx
activation_6/ReluReluadd_1/add:z:0*
T0*0
_output_shapes
:?????????  2
activation_6/ReluΎ
"block2/conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2/conv1/Conv2D/ReadVariableOpδ
block2/conv1/Conv2DConv2Dactivation_6/Relu:activations:0*block2/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/conv1/Conv2D΄
#block2/conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2/conv1/BiasAdd/ReadVariableOp½
block2/conv1/BiasAddBiasAddblock2/conv1/Conv2D:output:0+block2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/conv1/BiasAdd
activation_7/ReluRelublock2/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_7/ReluΣ
)block2/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block2_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block2/convshortcut/Conv2D/ReadVariableOpω
block2/convshortcut/Conv2DConv2Dactivation_6/Relu:activations:01block2/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/convshortcut/Conv2DΙ
*block2/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block2_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block2/convshortcut/BiasAdd/ReadVariableOpΩ
block2/convshortcut/BiasAddBiasAdd#block2/convshortcut/Conv2D:output:02block2/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/convshortcut/BiasAddΎ
"block2/conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2/conv2/Conv2D/ReadVariableOpδ
block2/conv2/Conv2DConv2Dactivation_7/Relu:activations:0*block2/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/conv2/Conv2D΄
#block2/conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2/conv2/BiasAdd/ReadVariableOp½
block2/conv2/BiasAddBiasAddblock2/conv2/Conv2D:output:0+block2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/conv2/BiasAdd
activation_8/ReluRelublock2/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_8/Relu
activation_9/ReluRelu$block2/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_9/Relu
	add_2/addAddV2activation_8/Relu:activations:0activation_9/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_2/addz
activation_10/ReluReluadd_2/add:z:0*
T0*0
_output_shapes
:?????????2
activation_10/ReluΎ
"block3/conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3/conv1/Conv2D/ReadVariableOpε
block3/conv1/Conv2DConv2D activation_10/Relu:activations:0*block3/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/conv1/Conv2D΄
#block3/conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3/conv1/BiasAdd/ReadVariableOp½
block3/conv1/BiasAddBiasAddblock3/conv1/Conv2D:output:0+block3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/conv1/BiasAdd
activation_11/ReluRelublock3/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_11/ReluΣ
)block3/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block3_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block3/convshortcut/Conv2D/ReadVariableOpϊ
block3/convshortcut/Conv2DConv2D activation_10/Relu:activations:01block3/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/convshortcut/Conv2DΙ
*block3/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block3_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block3/convshortcut/BiasAdd/ReadVariableOpΩ
block3/convshortcut/BiasAddBiasAdd#block3/convshortcut/Conv2D:output:02block3/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/convshortcut/BiasAddΎ
"block3/conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3/conv2/Conv2D/ReadVariableOpε
block3/conv2/Conv2DConv2D activation_11/Relu:activations:0*block3/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/conv2/Conv2D΄
#block3/conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3/conv2/BiasAdd/ReadVariableOp½
block3/conv2/BiasAddBiasAddblock3/conv2/Conv2D:output:0+block3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/conv2/BiasAdd
activation_12/ReluRelublock3/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_12/Relu
activation_13/ReluRelu$block3/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_13/Relu
	add_3/addAddV2 activation_12/Relu:activations:0 activation_13/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_3/addz
activation_14/ReluReluadd_3/add:z:0*
T0*0
_output_shapes
:?????????2
activation_14/ReluΎ
"block4/conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4/conv1/Conv2D/ReadVariableOpε
block4/conv1/Conv2DConv2D activation_14/Relu:activations:0*block4/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/conv1/Conv2D΄
#block4/conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4/conv1/BiasAdd/ReadVariableOp½
block4/conv1/BiasAddBiasAddblock4/conv1/Conv2D:output:0+block4/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/conv1/BiasAdd
activation_15/ReluRelublock4/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_15/ReluΣ
)block4/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block4_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block4/convshortcut/Conv2D/ReadVariableOpϊ
block4/convshortcut/Conv2DConv2D activation_14/Relu:activations:01block4/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/convshortcut/Conv2DΙ
*block4/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block4_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block4/convshortcut/BiasAdd/ReadVariableOpΩ
block4/convshortcut/BiasAddBiasAdd#block4/convshortcut/Conv2D:output:02block4/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/convshortcut/BiasAddΎ
"block4/conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4/conv2/Conv2D/ReadVariableOpε
block4/conv2/Conv2DConv2D activation_15/Relu:activations:0*block4/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/conv2/Conv2D΄
#block4/conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4/conv2/BiasAdd/ReadVariableOp½
block4/conv2/BiasAddBiasAddblock4/conv2/Conv2D:output:0+block4/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/conv2/BiasAdd
activation_16/ReluRelublock4/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_16/Relu
activation_17/ReluRelu$block4/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_17/Relu
	add_4/addAddV2 activation_16/Relu:activations:0 activation_17/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_4/addz
activation_18/ReluReluadd_4/add:z:0*
T0*0
_output_shapes
:?????????2
activation_18/ReluΎ
"block5/conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5/conv1/Conv2D/ReadVariableOpε
block5/conv1/Conv2DConv2D activation_18/Relu:activations:0*block5/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/conv1/Conv2D΄
#block5/conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5/conv1/BiasAdd/ReadVariableOp½
block5/conv1/BiasAddBiasAddblock5/conv1/Conv2D:output:0+block5/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/conv1/BiasAdd
activation_19/ReluRelublock5/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_19/ReluΣ
)block5/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block5_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block5/convshortcut/Conv2D/ReadVariableOpϊ
block5/convshortcut/Conv2DConv2D activation_18/Relu:activations:01block5/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/convshortcut/Conv2DΙ
*block5/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block5_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block5/convshortcut/BiasAdd/ReadVariableOpΩ
block5/convshortcut/BiasAddBiasAdd#block5/convshortcut/Conv2D:output:02block5/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/convshortcut/BiasAddΎ
"block5/conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5/conv2/Conv2D/ReadVariableOpε
block5/conv2/Conv2DConv2D activation_19/Relu:activations:0*block5/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/conv2/Conv2D΄
#block5/conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5/conv2/BiasAdd/ReadVariableOp½
block5/conv2/BiasAddBiasAddblock5/conv2/Conv2D:output:0+block5/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/conv2/BiasAdd
activation_20/ReluRelublock5/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_20/Relu
activation_21/ReluRelu$block5/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_21/Relu
	add_5/addAddV2 activation_20/Relu:activations:0 activation_21/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_5/addz
activation_22/ReluReluadd_5/add:z:0*
T0*0
_output_shapes
:?????????2
activation_22/ReluΎ
"block6/conv1/Conv2D/ReadVariableOpReadVariableOp+block6_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block6/conv1/Conv2D/ReadVariableOpε
block6/conv1/Conv2DConv2D activation_22/Relu:activations:0*block6/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/conv1/Conv2D΄
#block6/conv1/BiasAdd/ReadVariableOpReadVariableOp,block6_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block6/conv1/BiasAdd/ReadVariableOp½
block6/conv1/BiasAddBiasAddblock6/conv1/Conv2D:output:0+block6/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/conv1/BiasAdd
activation_23/ReluRelublock6/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_23/ReluΣ
)block6/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block6_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block6/convshortcut/Conv2D/ReadVariableOpϊ
block6/convshortcut/Conv2DConv2D activation_22/Relu:activations:01block6/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/convshortcut/Conv2DΙ
*block6/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block6_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block6/convshortcut/BiasAdd/ReadVariableOpΩ
block6/convshortcut/BiasAddBiasAdd#block6/convshortcut/Conv2D:output:02block6/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/convshortcut/BiasAddΎ
"block6/conv2/Conv2D/ReadVariableOpReadVariableOp+block6_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block6/conv2/Conv2D/ReadVariableOpε
block6/conv2/Conv2DConv2D activation_23/Relu:activations:0*block6/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/conv2/Conv2D΄
#block6/conv2/BiasAdd/ReadVariableOpReadVariableOp,block6_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block6/conv2/BiasAdd/ReadVariableOp½
block6/conv2/BiasAddBiasAddblock6/conv2/Conv2D:output:0+block6/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/conv2/BiasAdd
activation_24/ReluRelublock6/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_24/Relu
activation_25/ReluRelu$block6/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_25/Relu
	add_6/addAddV2 activation_24/Relu:activations:0 activation_25/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_6/addz
activation_26/ReluReluadd_6/add:z:0*
T0*0
_output_shapes
:?????????2
activation_26/Relu€
fc1/Tensordot/ReadVariableOpReadVariableOp%fc1_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc1/Tensordot/ReadVariableOpr
fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc1/Tensordot/axes}
fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc1/Tensordot/freez
fc1/Tensordot/ShapeShape activation_26/Relu:activations:0*
T0*
_output_shapes
:2
fc1/Tensordot/Shape|
fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/GatherV2/axisε
fc1/Tensordot/GatherV2GatherV2fc1/Tensordot/Shape:output:0fc1/Tensordot/free:output:0$fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc1/Tensordot/GatherV2
fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/GatherV2_1/axisλ
fc1/Tensordot/GatherV2_1GatherV2fc1/Tensordot/Shape:output:0fc1/Tensordot/axes:output:0&fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc1/Tensordot/GatherV2_1t
fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc1/Tensordot/Const
fc1/Tensordot/ProdProdfc1/Tensordot/GatherV2:output:0fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc1/Tensordot/Prodx
fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc1/Tensordot/Const_1
fc1/Tensordot/Prod_1Prod!fc1/Tensordot/GatherV2_1:output:0fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc1/Tensordot/Prod_1x
fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/concat/axisΔ
fc1/Tensordot/concatConcatV2fc1/Tensordot/free:output:0fc1/Tensordot/axes:output:0"fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/concat
fc1/Tensordot/stackPackfc1/Tensordot/Prod:output:0fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/stack»
fc1/Tensordot/transpose	Transpose activation_26/Relu:activations:0fc1/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc1/Tensordot/transpose―
fc1/Tensordot/ReshapeReshapefc1/Tensordot/transpose:y:0fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc1/Tensordot/Reshape―
fc1/Tensordot/MatMulMatMulfc1/Tensordot/Reshape:output:0$fc1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc1/Tensordot/MatMuly
fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc1/Tensordot/Const_2|
fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/concat_1/axisΡ
fc1/Tensordot/concat_1ConcatV2fc1/Tensordot/GatherV2:output:0fc1/Tensordot/Const_2:output:0$fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/concat_1₯
fc1/TensordotReshapefc1/Tensordot/MatMul:product:0fc1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc1/Tensordot
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/Tensordot:output:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc1/BiasAddm
fc1/ReluRelufc1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc1/Relu€
fc2/Tensordot/ReadVariableOpReadVariableOp%fc2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/Tensordot/ReadVariableOpr
fc2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc2/Tensordot/axes}
fc2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc2/Tensordot/freep
fc2/Tensordot/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:2
fc2/Tensordot/Shape|
fc2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/GatherV2/axisε
fc2/Tensordot/GatherV2GatherV2fc2/Tensordot/Shape:output:0fc2/Tensordot/free:output:0$fc2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc2/Tensordot/GatherV2
fc2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/GatherV2_1/axisλ
fc2/Tensordot/GatherV2_1GatherV2fc2/Tensordot/Shape:output:0fc2/Tensordot/axes:output:0&fc2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc2/Tensordot/GatherV2_1t
fc2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc2/Tensordot/Const
fc2/Tensordot/ProdProdfc2/Tensordot/GatherV2:output:0fc2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc2/Tensordot/Prodx
fc2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc2/Tensordot/Const_1
fc2/Tensordot/Prod_1Prod!fc2/Tensordot/GatherV2_1:output:0fc2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc2/Tensordot/Prod_1x
fc2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/concat/axisΔ
fc2/Tensordot/concatConcatV2fc2/Tensordot/free:output:0fc2/Tensordot/axes:output:0"fc2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/concat
fc2/Tensordot/stackPackfc2/Tensordot/Prod:output:0fc2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/stack±
fc2/Tensordot/transpose	Transposefc1/Relu:activations:0fc2/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc2/Tensordot/transpose―
fc2/Tensordot/ReshapeReshapefc2/Tensordot/transpose:y:0fc2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc2/Tensordot/Reshape―
fc2/Tensordot/MatMulMatMulfc2/Tensordot/Reshape:output:0$fc2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc2/Tensordot/MatMuly
fc2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc2/Tensordot/Const_2|
fc2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/concat_1/axisΡ
fc2/Tensordot/concat_1ConcatV2fc2/Tensordot/GatherV2:output:0fc2/Tensordot/Const_2:output:0$fc2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/concat_1₯
fc2/TensordotReshapefc2/Tensordot/MatMul:product:0fc2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc2/Tensordot
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/Tensordot:output:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc2/BiasAddm
fc2/ReluRelufc2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc2/Relu€
fc3/Tensordot/ReadVariableOpReadVariableOp%fc3_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc3/Tensordot/ReadVariableOpr
fc3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc3/Tensordot/axes}
fc3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc3/Tensordot/freep
fc3/Tensordot/ShapeShapefc2/Relu:activations:0*
T0*
_output_shapes
:2
fc3/Tensordot/Shape|
fc3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/GatherV2/axisε
fc3/Tensordot/GatherV2GatherV2fc3/Tensordot/Shape:output:0fc3/Tensordot/free:output:0$fc3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc3/Tensordot/GatherV2
fc3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/GatherV2_1/axisλ
fc3/Tensordot/GatherV2_1GatherV2fc3/Tensordot/Shape:output:0fc3/Tensordot/axes:output:0&fc3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc3/Tensordot/GatherV2_1t
fc3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc3/Tensordot/Const
fc3/Tensordot/ProdProdfc3/Tensordot/GatherV2:output:0fc3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc3/Tensordot/Prodx
fc3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc3/Tensordot/Const_1
fc3/Tensordot/Prod_1Prod!fc3/Tensordot/GatherV2_1:output:0fc3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc3/Tensordot/Prod_1x
fc3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/concat/axisΔ
fc3/Tensordot/concatConcatV2fc3/Tensordot/free:output:0fc3/Tensordot/axes:output:0"fc3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/concat
fc3/Tensordot/stackPackfc3/Tensordot/Prod:output:0fc3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/stack±
fc3/Tensordot/transpose	Transposefc2/Relu:activations:0fc3/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc3/Tensordot/transpose―
fc3/Tensordot/ReshapeReshapefc3/Tensordot/transpose:y:0fc3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc3/Tensordot/Reshape―
fc3/Tensordot/MatMulMatMulfc3/Tensordot/Reshape:output:0$fc3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc3/Tensordot/MatMuly
fc3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc3/Tensordot/Const_2|
fc3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/concat_1/axisΡ
fc3/Tensordot/concat_1ConcatV2fc3/Tensordot/GatherV2:output:0fc3/Tensordot/Const_2:output:0$fc3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/concat_1₯
fc3/TensordotReshapefc3/Tensordot/MatMul:product:0fc3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc3/Tensordot
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/Tensordot:output:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc3/BiasAddm
fc3/ReluRelufc3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc3/Relu¬
fc_out/Tensordot/ReadVariableOpReadVariableOp(fc_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02!
fc_out/Tensordot/ReadVariableOpx
fc_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc_out/Tensordot/axes
fc_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_out/Tensordot/freev
fc_out/Tensordot/ShapeShapefc3/Relu:activations:0*
T0*
_output_shapes
:2
fc_out/Tensordot/Shape
fc_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
fc_out/Tensordot/GatherV2/axisτ
fc_out/Tensordot/GatherV2GatherV2fc_out/Tensordot/Shape:output:0fc_out/Tensordot/free:output:0'fc_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc_out/Tensordot/GatherV2
 fc_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 fc_out/Tensordot/GatherV2_1/axisϊ
fc_out/Tensordot/GatherV2_1GatherV2fc_out/Tensordot/Shape:output:0fc_out/Tensordot/axes:output:0)fc_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc_out/Tensordot/GatherV2_1z
fc_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc_out/Tensordot/Const
fc_out/Tensordot/ProdProd"fc_out/Tensordot/GatherV2:output:0fc_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc_out/Tensordot/Prod~
fc_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc_out/Tensordot/Const_1€
fc_out/Tensordot/Prod_1Prod$fc_out/Tensordot/GatherV2_1:output:0!fc_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc_out/Tensordot/Prod_1~
fc_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_out/Tensordot/concat/axisΣ
fc_out/Tensordot/concatConcatV2fc_out/Tensordot/free:output:0fc_out/Tensordot/axes:output:0%fc_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/concat¨
fc_out/Tensordot/stackPackfc_out/Tensordot/Prod:output:0 fc_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/stackΊ
fc_out/Tensordot/transpose	Transposefc3/Relu:activations:0 fc_out/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc_out/Tensordot/transpose»
fc_out/Tensordot/ReshapeReshapefc_out/Tensordot/transpose:y:0fc_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc_out/Tensordot/ReshapeΊ
fc_out/Tensordot/MatMulMatMul!fc_out/Tensordot/Reshape:output:0'fc_out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fc_out/Tensordot/MatMul~
fc_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc_out/Tensordot/Const_2
fc_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
fc_out/Tensordot/concat_1/axisΰ
fc_out/Tensordot/concat_1ConcatV2"fc_out/Tensordot/GatherV2:output:0!fc_out/Tensordot/Const_2:output:0'fc_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/concat_1°
fc_out/TensordotReshape!fc_out/Tensordot/MatMul:product:0"fc_out/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????2
fc_out/Tensordot‘
fc_out/BiasAdd/ReadVariableOpReadVariableOp&fc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_out/BiasAdd/ReadVariableOp§
fc_out/BiasAddBiasAddfc_out/Tensordot:output:0%fc_out/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
fc_out/BiasAdd~
fc_out/SigmoidSigmoidfc_out/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
fc_out/Sigmoidn
IdentityIdentityfc_out/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H:::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Ή
Έ
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_3745592

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
S
'__inference_add_2_layer_call_fn_3747687
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_37450862
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Ξ
S
'__inference_add_4_layer_call_fn_3747905
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_4_layer_call_and_return_conditional_losses_37453762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_11_layer_call_and_return_conditional_losses_3747721

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_6_layer_call_and_return_conditional_losses_3747583

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_11_layer_call_and_return_conditional_losses_3745139

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_11_layer_call_fn_3747726

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_11_layer_call_and_return_conditional_losses_37451392
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block5/conv2_layer_call_and_return_conditional_losses_3747954

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_20_layer_call_and_return_conditional_losses_3745494

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block2/conv2_layer_call_and_return_conditional_losses_3747627

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_3747646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ή
n
B__inference_add_4_layer_call_and_return_conditional_losses_3747899
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_18_layer_call_and_return_conditional_losses_3747910

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ
Q
%__inference_add_layer_call_fn_3747498
inputs_0
inputs_1
identityΩ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_37448352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :Z V
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
Φ
l
B__inference_add_6_layer_call_and_return_conditional_losses_3745666

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ρ
ί
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746035
input_1
conv01_3745880
conv01_3745882
block0_conv1_3745886
block0_conv1_3745888
block0_conv2_3745892
block0_conv2_3745894
block1_conv1_3745900
block1_conv1_3745902
block1_conv2_3745906
block1_conv2_3745908
block2_conv1_3745914
block2_conv1_3745916
block2_convshortcut_3745920
block2_convshortcut_3745922
block2_conv2_3745925
block2_conv2_3745927
block3_conv1_3745934
block3_conv1_3745936
block3_convshortcut_3745940
block3_convshortcut_3745942
block3_conv2_3745945
block3_conv2_3745947
block4_conv1_3745954
block4_conv1_3745956
block4_convshortcut_3745960
block4_convshortcut_3745962
block4_conv2_3745965
block4_conv2_3745967
block5_conv1_3745974
block5_conv1_3745976
block5_convshortcut_3745980
block5_convshortcut_3745982
block5_conv2_3745985
block5_conv2_3745987
block6_conv1_3745994
block6_conv1_3745996
block6_convshortcut_3746000
block6_convshortcut_3746002
block6_conv2_3746005
block6_conv2_3746007
fc1_3746014
fc1_3746016
fc2_3746019
fc2_3746021
fc3_3746024
fc3_3746026
fc_out_3746029
fc_out_3746031
identity’$block0/conv1/StatefulPartitionedCall’$block0/conv2/StatefulPartitionedCall’$block1/conv1/StatefulPartitionedCall’$block1/conv2/StatefulPartitionedCall’$block2/conv1/StatefulPartitionedCall’$block2/conv2/StatefulPartitionedCall’+block2/convshortcut/StatefulPartitionedCall’$block3/conv1/StatefulPartitionedCall’$block3/conv2/StatefulPartitionedCall’+block3/convshortcut/StatefulPartitionedCall’$block4/conv1/StatefulPartitionedCall’$block4/conv2/StatefulPartitionedCall’+block4/convshortcut/StatefulPartitionedCall’$block5/conv1/StatefulPartitionedCall’$block5/conv2/StatefulPartitionedCall’+block5/convshortcut/StatefulPartitionedCall’$block6/conv1/StatefulPartitionedCall’$block6/conv2/StatefulPartitionedCall’+block6/convshortcut/StatefulPartitionedCall’conv01/StatefulPartitionedCall’fc1/StatefulPartitionedCall’fc2/StatefulPartitionedCall’fc3/StatefulPartitionedCall’fc_out/StatefulPartitionedCall
conv01/StatefulPartitionedCallStatefulPartitionedCallinput_1conv01_3745880conv01_3745882*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv01_layer_call_and_return_conditional_losses_37447222 
conv01/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_37447432
activation/PartitionedCallΦ
$block0/conv1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0block0_conv1_3745886block0_conv1_3745888*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv1_layer_call_and_return_conditional_losses_37447612&
$block0/conv1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall-block0/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_37447822
activation_1/PartitionedCallΨ
$block0/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0block0_conv2_3745892block0_conv2_3745894*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv2_layer_call_and_return_conditional_losses_37448002&
$block0/conv2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall-block0/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_37448212
activation_2/PartitionedCall
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_37448352
add/PartitionedCall
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_37448492
activation_3/PartitionedCallΨ
$block1/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0block1_conv1_3745900block1_conv1_3745902*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv1_layer_call_and_return_conditional_losses_37448672&
$block1/conv1/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall-block1/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_37448882
activation_4/PartitionedCallΨ
$block1/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0block1_conv2_3745906block1_conv2_3745908*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv2_layer_call_and_return_conditional_losses_37449062&
$block1/conv2/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall-block1/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_37449272
activation_5/PartitionedCall‘
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_37449412
add_1/PartitionedCall
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_37449552
activation_6/PartitionedCallΨ
$block2/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_conv1_3745914block2_conv1_3745916*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv1_layer_call_and_return_conditional_losses_37449732&
$block2/conv1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall-block2/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_37449942
activation_7/PartitionedCallϋ
+block2/convshortcut/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_convshortcut_3745920block2_convshortcut_3745922*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_37450122-
+block2/convshortcut/StatefulPartitionedCallΨ
$block2/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0block2_conv2_3745925block2_conv2_3745927*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv2_layer_call_and_return_conditional_losses_37450382&
$block2/conv2/StatefulPartitionedCall
activation_8/PartitionedCallPartitionedCall-block2/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_8_layer_call_and_return_conditional_losses_37450592
activation_8/PartitionedCall
activation_9/PartitionedCallPartitionedCall4block2/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_9_layer_call_and_return_conditional_losses_37450722
activation_9/PartitionedCall‘
add_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_37450862
add_2/PartitionedCall
activation_10/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_10_layer_call_and_return_conditional_losses_37451002
activation_10/PartitionedCallΩ
$block3/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_conv1_3745934block3_conv1_3745936*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv1_layer_call_and_return_conditional_losses_37451182&
$block3/conv1/StatefulPartitionedCall
activation_11/PartitionedCallPartitionedCall-block3/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_11_layer_call_and_return_conditional_losses_37451392
activation_11/PartitionedCallό
+block3/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_convshortcut_3745940block3_convshortcut_3745942*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_37451572-
+block3/convshortcut/StatefulPartitionedCallΩ
$block3/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_11/PartitionedCall:output:0block3_conv2_3745945block3_conv2_3745947*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv2_layer_call_and_return_conditional_losses_37451832&
$block3/conv2/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall-block3/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_12_layer_call_and_return_conditional_losses_37452042
activation_12/PartitionedCall 
activation_13/PartitionedCallPartitionedCall4block3/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_13_layer_call_and_return_conditional_losses_37452172
activation_13/PartitionedCall£
add_3/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_37452312
add_3/PartitionedCall
activation_14/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_14_layer_call_and_return_conditional_losses_37452452
activation_14/PartitionedCallΩ
$block4/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_conv1_3745954block4_conv1_3745956*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv1_layer_call_and_return_conditional_losses_37452632&
$block4/conv1/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall-block4/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_15_layer_call_and_return_conditional_losses_37452842
activation_15/PartitionedCallό
+block4/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_convshortcut_3745960block4_convshortcut_3745962*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_37453022-
+block4/convshortcut/StatefulPartitionedCallΩ
$block4/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0block4_conv2_3745965block4_conv2_3745967*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv2_layer_call_and_return_conditional_losses_37453282&
$block4/conv2/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall-block4/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_16_layer_call_and_return_conditional_losses_37453492
activation_16/PartitionedCall 
activation_17/PartitionedCallPartitionedCall4block4/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_17_layer_call_and_return_conditional_losses_37453622
activation_17/PartitionedCall£
add_4/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_4_layer_call_and_return_conditional_losses_37453762
add_4/PartitionedCall
activation_18/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_18_layer_call_and_return_conditional_losses_37453902
activation_18/PartitionedCallΩ
$block5/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_conv1_3745974block5_conv1_3745976*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv1_layer_call_and_return_conditional_losses_37454082&
$block5/conv1/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-block5/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_19_layer_call_and_return_conditional_losses_37454292
activation_19/PartitionedCallό
+block5/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_convshortcut_3745980block5_convshortcut_3745982*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_37454472-
+block5/convshortcut/StatefulPartitionedCallΩ
$block5/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0block5_conv2_3745985block5_conv2_3745987*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv2_layer_call_and_return_conditional_losses_37454732&
$block5/conv2/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall-block5/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_20_layer_call_and_return_conditional_losses_37454942
activation_20/PartitionedCall 
activation_21/PartitionedCallPartitionedCall4block5/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_21_layer_call_and_return_conditional_losses_37455072
activation_21/PartitionedCall£
add_5/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_5_layer_call_and_return_conditional_losses_37455212
add_5/PartitionedCall
activation_22/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_22_layer_call_and_return_conditional_losses_37455352
activation_22/PartitionedCallΩ
$block6/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_conv1_3745994block6_conv1_3745996*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv1_layer_call_and_return_conditional_losses_37455532&
$block6/conv1/StatefulPartitionedCall
activation_23/PartitionedCallPartitionedCall-block6/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_23_layer_call_and_return_conditional_losses_37455742
activation_23/PartitionedCallό
+block6/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_convshortcut_3746000block6_convshortcut_3746002*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_37455922-
+block6/convshortcut/StatefulPartitionedCallΩ
$block6/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0block6_conv2_3746005block6_conv2_3746007*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv2_layer_call_and_return_conditional_losses_37456182&
$block6/conv2/StatefulPartitionedCall
activation_24/PartitionedCallPartitionedCall-block6/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_24_layer_call_and_return_conditional_losses_37456392
activation_24/PartitionedCall 
activation_25/PartitionedCallPartitionedCall4block6/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_25_layer_call_and_return_conditional_losses_37456522
activation_25/PartitionedCall£
add_6/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_6_layer_call_and_return_conditional_losses_37456662
add_6/PartitionedCall
activation_26/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_26_layer_call_and_return_conditional_losses_37456802
activation_26/PartitionedCall¬
fc1/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0fc1_3746014fc1_3746016*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_37457192
fc1/StatefulPartitionedCallͺ
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0fc2_3746019fc2_3746021*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_37457662
fc2/StatefulPartitionedCallͺ
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0fc3_3746024fc3_3746026*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc3_layer_call_and_return_conditional_losses_37458132
fc3/StatefulPartitionedCallΈ
fc_out/StatefulPartitionedCallStatefulPartitionedCall$fc3/StatefulPartitionedCall:output:0fc_out_3746029fc_out_3746031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_fc_out_layer_call_and_return_conditional_losses_37458602 
fc_out/StatefulPartitionedCall§
IdentityIdentity'fc_out/StatefulPartitionedCall:output:0%^block0/conv1/StatefulPartitionedCall%^block0/conv2/StatefulPartitionedCall%^block1/conv1/StatefulPartitionedCall%^block1/conv2/StatefulPartitionedCall%^block2/conv1/StatefulPartitionedCall%^block2/conv2/StatefulPartitionedCall,^block2/convshortcut/StatefulPartitionedCall%^block3/conv1/StatefulPartitionedCall%^block3/conv2/StatefulPartitionedCall,^block3/convshortcut/StatefulPartitionedCall%^block4/conv1/StatefulPartitionedCall%^block4/conv2/StatefulPartitionedCall,^block4/convshortcut/StatefulPartitionedCall%^block5/conv1/StatefulPartitionedCall%^block5/conv2/StatefulPartitionedCall,^block5/convshortcut/StatefulPartitionedCall%^block6/conv1/StatefulPartitionedCall%^block6/conv2/StatefulPartitionedCall,^block6/convshortcut/StatefulPartitionedCall^conv01/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall^fc_out/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::2L
$block0/conv1/StatefulPartitionedCall$block0/conv1/StatefulPartitionedCall2L
$block0/conv2/StatefulPartitionedCall$block0/conv2/StatefulPartitionedCall2L
$block1/conv1/StatefulPartitionedCall$block1/conv1/StatefulPartitionedCall2L
$block1/conv2/StatefulPartitionedCall$block1/conv2/StatefulPartitionedCall2L
$block2/conv1/StatefulPartitionedCall$block2/conv1/StatefulPartitionedCall2L
$block2/conv2/StatefulPartitionedCall$block2/conv2/StatefulPartitionedCall2Z
+block2/convshortcut/StatefulPartitionedCall+block2/convshortcut/StatefulPartitionedCall2L
$block3/conv1/StatefulPartitionedCall$block3/conv1/StatefulPartitionedCall2L
$block3/conv2/StatefulPartitionedCall$block3/conv2/StatefulPartitionedCall2Z
+block3/convshortcut/StatefulPartitionedCall+block3/convshortcut/StatefulPartitionedCall2L
$block4/conv1/StatefulPartitionedCall$block4/conv1/StatefulPartitionedCall2L
$block4/conv2/StatefulPartitionedCall$block4/conv2/StatefulPartitionedCall2Z
+block4/convshortcut/StatefulPartitionedCall+block4/convshortcut/StatefulPartitionedCall2L
$block5/conv1/StatefulPartitionedCall$block5/conv1/StatefulPartitionedCall2L
$block5/conv2/StatefulPartitionedCall$block5/conv2/StatefulPartitionedCall2Z
+block5/convshortcut/StatefulPartitionedCall+block5/convshortcut/StatefulPartitionedCall2L
$block6/conv1/StatefulPartitionedCall$block6/conv1/StatefulPartitionedCall2L
$block6/conv2/StatefulPartitionedCall$block6/conv2/StatefulPartitionedCall2Z
+block6/convshortcut/StatefulPartitionedCall+block6/convshortcut/StatefulPartitionedCall2@
conv01/StatefulPartitionedCallconv01/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall2@
fc_out/StatefulPartitionedCallfc_out/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
ά
f
J__inference_activation_17_layer_call_and_return_conditional_losses_3745362

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_24_layer_call_and_return_conditional_losses_3745639

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_3745447

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


5__inference_block3/convshortcut_layer_call_fn_3747764

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_37451572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_23_layer_call_fn_3748053

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_23_layer_call_and_return_conditional_losses_37455742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
μ
?
%__inference_signature_wrapper_3746665
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_37447082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
ά
f
J__inference_activation_23_layer_call_and_return_conditional_losses_3745574

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_20_layer_call_and_return_conditional_losses_3747987

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block4/conv2_layer_call_and_return_conditional_losses_3745328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

Υ
(__inference_MTFNet_layer_call_fn_3746554
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_MTFNet_layer_call_and_return_conditional_losses_37464552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
Ϋ
e
I__inference_activation_5_layer_call_and_return_conditional_losses_3747561

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ε
J
.__inference_activation_6_layer_call_fn_3747588

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_37449552
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs

«
@__inference_fc3_layer_call_and_return_conditional_losses_3745813

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_3747755

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
S
'__inference_add_3_layer_call_fn_3747796
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_37452312
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_25_layer_call_and_return_conditional_losses_3748106

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
S
'__inference_add_5_layer_call_fn_3748014
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_5_layer_call_and_return_conditional_losses_37455212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_10_layer_call_and_return_conditional_losses_3747692

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


5__inference_block2/convshortcut_layer_call_fn_3747655

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_37450122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Η
K
/__inference_activation_20_layer_call_fn_3747992

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_20_layer_call_and_return_conditional_losses_37454942
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

?
C__inference_fc_out_layer_call_and_return_conditional_losses_3748284

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_22_layer_call_and_return_conditional_losses_3745535

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

}
(__inference_fc_out_layer_call_fn_3748293

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_fc_out_layer_call_and_return_conditional_losses_37458602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

Τ
(__inference_MTFNet_layer_call_fn_3747399

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_MTFNet_layer_call_and_return_conditional_losses_37464552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Ή
Έ
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_3745157

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

«
@__inference_fc2_layer_call_and_return_conditional_losses_3745766

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block0/conv2_layer_call_fn_3747476

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv2_layer_call_and_return_conditional_losses_37448002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ή
n
B__inference_add_5_layer_call_and_return_conditional_losses_3748008
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1


.__inference_block5/conv2_layer_call_fn_3747963

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv2_layer_call_and_return_conditional_losses_37454732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Τ
j
@__inference_add_layer_call_and_return_conditional_losses_3744835

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????  2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
²
±
I__inference_block5/conv1_layer_call_and_return_conditional_losses_3747925

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_4_layer_call_and_return_conditional_losses_3747532

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Φ
l
B__inference_add_1_layer_call_and_return_conditional_losses_3744941

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????  2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????  
 
_user_specified_nameinputs

«
@__inference_fc1_layer_call_and_return_conditional_losses_3745719

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
·
?
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746931

inputs)
%conv01_conv2d_readvariableop_resource*
&conv01_biasadd_readvariableop_resource/
+block0_conv1_conv2d_readvariableop_resource0
,block0_conv1_biasadd_readvariableop_resource/
+block0_conv2_conv2d_readvariableop_resource0
,block0_conv2_biasadd_readvariableop_resource/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource6
2block2_convshortcut_conv2d_readvariableop_resource7
3block2_convshortcut_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource6
2block3_convshortcut_conv2d_readvariableop_resource7
3block3_convshortcut_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource6
2block4_convshortcut_conv2d_readvariableop_resource7
3block4_convshortcut_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource6
2block5_convshortcut_conv2d_readvariableop_resource7
3block5_convshortcut_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block6_conv1_conv2d_readvariableop_resource0
,block6_conv1_biasadd_readvariableop_resource6
2block6_convshortcut_conv2d_readvariableop_resource7
3block6_convshortcut_biasadd_readvariableop_resource/
+block6_conv2_conv2d_readvariableop_resource0
,block6_conv2_biasadd_readvariableop_resource)
%fc1_tensordot_readvariableop_resource'
#fc1_biasadd_readvariableop_resource)
%fc2_tensordot_readvariableop_resource'
#fc2_biasadd_readvariableop_resource)
%fc3_tensordot_readvariableop_resource'
#fc3_biasadd_readvariableop_resource,
(fc_out_tensordot_readvariableop_resource*
&fc_out_biasadd_readvariableop_resource
identity«
conv01/Conv2D/ReadVariableOpReadVariableOp%conv01_conv2d_readvariableop_resource*'
_output_shapes
:H*
dtype02
conv01/Conv2D/ReadVariableOpΉ
conv01/Conv2DConv2Dinputs$conv01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv01/Conv2D’
conv01/BiasAdd/ReadVariableOpReadVariableOp&conv01_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv01/BiasAdd/ReadVariableOp₯
conv01/BiasAddBiasAddconv01/Conv2D:output:0%conv01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
conv01/BiasAdd~
activation/ReluReluconv01/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation/ReluΎ
"block0/conv1/Conv2D/ReadVariableOpReadVariableOp+block0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block0/conv1/Conv2D/ReadVariableOpβ
block0/conv1/Conv2DConv2Dactivation/Relu:activations:0*block0/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block0/conv1/Conv2D΄
#block0/conv1/BiasAdd/ReadVariableOpReadVariableOp,block0_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block0/conv1/BiasAdd/ReadVariableOp½
block0/conv1/BiasAddBiasAddblock0/conv1/Conv2D:output:0+block0/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block0/conv1/BiasAdd
activation_1/ReluRelublock0/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_1/ReluΎ
"block0/conv2/Conv2D/ReadVariableOpReadVariableOp+block0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block0/conv2/Conv2D/ReadVariableOpδ
block0/conv2/Conv2DConv2Dactivation_1/Relu:activations:0*block0/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block0/conv2/Conv2D΄
#block0/conv2/BiasAdd/ReadVariableOpReadVariableOp,block0_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block0/conv2/BiasAdd/ReadVariableOp½
block0/conv2/BiasAddBiasAddblock0/conv2/Conv2D:output:0+block0/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block0/conv2/BiasAdd
activation_2/ReluRelublock0/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_2/Relu
add/addAddV2activation_2/Relu:activations:0activation/Relu:activations:0*
T0*0
_output_shapes
:?????????  2	
add/addv
activation_3/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????  2
activation_3/ReluΎ
"block1/conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block1/conv1/Conv2D/ReadVariableOpδ
block1/conv1/Conv2DConv2Dactivation_3/Relu:activations:0*block1/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block1/conv1/Conv2D΄
#block1/conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block1/conv1/BiasAdd/ReadVariableOp½
block1/conv1/BiasAddBiasAddblock1/conv1/Conv2D:output:0+block1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block1/conv1/BiasAdd
activation_4/ReluRelublock1/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_4/ReluΎ
"block1/conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block1/conv2/Conv2D/ReadVariableOpδ
block1/conv2/Conv2DConv2Dactivation_4/Relu:activations:0*block1/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
block1/conv2/Conv2D΄
#block1/conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block1/conv2/BiasAdd/ReadVariableOp½
block1/conv2/BiasAddBiasAddblock1/conv2/Conv2D:output:0+block1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
block1/conv2/BiasAdd
activation_5/ReluRelublock1/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
activation_5/Relu
	add_1/addAddV2activation_5/Relu:activations:0activation_3/Relu:activations:0*
T0*0
_output_shapes
:?????????  2
	add_1/addx
activation_6/ReluReluadd_1/add:z:0*
T0*0
_output_shapes
:?????????  2
activation_6/ReluΎ
"block2/conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2/conv1/Conv2D/ReadVariableOpδ
block2/conv1/Conv2DConv2Dactivation_6/Relu:activations:0*block2/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/conv1/Conv2D΄
#block2/conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2/conv1/BiasAdd/ReadVariableOp½
block2/conv1/BiasAddBiasAddblock2/conv1/Conv2D:output:0+block2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/conv1/BiasAdd
activation_7/ReluRelublock2/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_7/ReluΣ
)block2/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block2_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block2/convshortcut/Conv2D/ReadVariableOpω
block2/convshortcut/Conv2DConv2Dactivation_6/Relu:activations:01block2/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/convshortcut/Conv2DΙ
*block2/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block2_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block2/convshortcut/BiasAdd/ReadVariableOpΩ
block2/convshortcut/BiasAddBiasAdd#block2/convshortcut/Conv2D:output:02block2/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/convshortcut/BiasAddΎ
"block2/conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2/conv2/Conv2D/ReadVariableOpδ
block2/conv2/Conv2DConv2Dactivation_7/Relu:activations:0*block2/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block2/conv2/Conv2D΄
#block2/conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2/conv2/BiasAdd/ReadVariableOp½
block2/conv2/BiasAddBiasAddblock2/conv2/Conv2D:output:0+block2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block2/conv2/BiasAdd
activation_8/ReluRelublock2/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_8/Relu
activation_9/ReluRelu$block2/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_9/Relu
	add_2/addAddV2activation_8/Relu:activations:0activation_9/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_2/addz
activation_10/ReluReluadd_2/add:z:0*
T0*0
_output_shapes
:?????????2
activation_10/ReluΎ
"block3/conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3/conv1/Conv2D/ReadVariableOpε
block3/conv1/Conv2DConv2D activation_10/Relu:activations:0*block3/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/conv1/Conv2D΄
#block3/conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3/conv1/BiasAdd/ReadVariableOp½
block3/conv1/BiasAddBiasAddblock3/conv1/Conv2D:output:0+block3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/conv1/BiasAdd
activation_11/ReluRelublock3/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_11/ReluΣ
)block3/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block3_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block3/convshortcut/Conv2D/ReadVariableOpϊ
block3/convshortcut/Conv2DConv2D activation_10/Relu:activations:01block3/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/convshortcut/Conv2DΙ
*block3/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block3_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block3/convshortcut/BiasAdd/ReadVariableOpΩ
block3/convshortcut/BiasAddBiasAdd#block3/convshortcut/Conv2D:output:02block3/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/convshortcut/BiasAddΎ
"block3/conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3/conv2/Conv2D/ReadVariableOpε
block3/conv2/Conv2DConv2D activation_11/Relu:activations:0*block3/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block3/conv2/Conv2D΄
#block3/conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3/conv2/BiasAdd/ReadVariableOp½
block3/conv2/BiasAddBiasAddblock3/conv2/Conv2D:output:0+block3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block3/conv2/BiasAdd
activation_12/ReluRelublock3/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_12/Relu
activation_13/ReluRelu$block3/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_13/Relu
	add_3/addAddV2 activation_12/Relu:activations:0 activation_13/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_3/addz
activation_14/ReluReluadd_3/add:z:0*
T0*0
_output_shapes
:?????????2
activation_14/ReluΎ
"block4/conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4/conv1/Conv2D/ReadVariableOpε
block4/conv1/Conv2DConv2D activation_14/Relu:activations:0*block4/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/conv1/Conv2D΄
#block4/conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4/conv1/BiasAdd/ReadVariableOp½
block4/conv1/BiasAddBiasAddblock4/conv1/Conv2D:output:0+block4/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/conv1/BiasAdd
activation_15/ReluRelublock4/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_15/ReluΣ
)block4/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block4_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block4/convshortcut/Conv2D/ReadVariableOpϊ
block4/convshortcut/Conv2DConv2D activation_14/Relu:activations:01block4/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/convshortcut/Conv2DΙ
*block4/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block4_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block4/convshortcut/BiasAdd/ReadVariableOpΩ
block4/convshortcut/BiasAddBiasAdd#block4/convshortcut/Conv2D:output:02block4/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/convshortcut/BiasAddΎ
"block4/conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4/conv2/Conv2D/ReadVariableOpε
block4/conv2/Conv2DConv2D activation_15/Relu:activations:0*block4/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block4/conv2/Conv2D΄
#block4/conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4/conv2/BiasAdd/ReadVariableOp½
block4/conv2/BiasAddBiasAddblock4/conv2/Conv2D:output:0+block4/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block4/conv2/BiasAdd
activation_16/ReluRelublock4/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_16/Relu
activation_17/ReluRelu$block4/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_17/Relu
	add_4/addAddV2 activation_16/Relu:activations:0 activation_17/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_4/addz
activation_18/ReluReluadd_4/add:z:0*
T0*0
_output_shapes
:?????????2
activation_18/ReluΎ
"block5/conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5/conv1/Conv2D/ReadVariableOpε
block5/conv1/Conv2DConv2D activation_18/Relu:activations:0*block5/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/conv1/Conv2D΄
#block5/conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5/conv1/BiasAdd/ReadVariableOp½
block5/conv1/BiasAddBiasAddblock5/conv1/Conv2D:output:0+block5/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/conv1/BiasAdd
activation_19/ReluRelublock5/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_19/ReluΣ
)block5/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block5_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block5/convshortcut/Conv2D/ReadVariableOpϊ
block5/convshortcut/Conv2DConv2D activation_18/Relu:activations:01block5/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/convshortcut/Conv2DΙ
*block5/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block5_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block5/convshortcut/BiasAdd/ReadVariableOpΩ
block5/convshortcut/BiasAddBiasAdd#block5/convshortcut/Conv2D:output:02block5/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/convshortcut/BiasAddΎ
"block5/conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5/conv2/Conv2D/ReadVariableOpε
block5/conv2/Conv2DConv2D activation_19/Relu:activations:0*block5/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block5/conv2/Conv2D΄
#block5/conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5/conv2/BiasAdd/ReadVariableOp½
block5/conv2/BiasAddBiasAddblock5/conv2/Conv2D:output:0+block5/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block5/conv2/BiasAdd
activation_20/ReluRelublock5/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_20/Relu
activation_21/ReluRelu$block5/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_21/Relu
	add_5/addAddV2 activation_20/Relu:activations:0 activation_21/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_5/addz
activation_22/ReluReluadd_5/add:z:0*
T0*0
_output_shapes
:?????????2
activation_22/ReluΎ
"block6/conv1/Conv2D/ReadVariableOpReadVariableOp+block6_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block6/conv1/Conv2D/ReadVariableOpε
block6/conv1/Conv2DConv2D activation_22/Relu:activations:0*block6/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/conv1/Conv2D΄
#block6/conv1/BiasAdd/ReadVariableOpReadVariableOp,block6_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block6/conv1/BiasAdd/ReadVariableOp½
block6/conv1/BiasAddBiasAddblock6/conv1/Conv2D:output:0+block6/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/conv1/BiasAdd
activation_23/ReluRelublock6/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_23/ReluΣ
)block6/convshortcut/Conv2D/ReadVariableOpReadVariableOp2block6_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)block6/convshortcut/Conv2D/ReadVariableOpϊ
block6/convshortcut/Conv2DConv2D activation_22/Relu:activations:01block6/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/convshortcut/Conv2DΙ
*block6/convshortcut/BiasAdd/ReadVariableOpReadVariableOp3block6_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*block6/convshortcut/BiasAdd/ReadVariableOpΩ
block6/convshortcut/BiasAddBiasAdd#block6/convshortcut/Conv2D:output:02block6/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/convshortcut/BiasAddΎ
"block6/conv2/Conv2D/ReadVariableOpReadVariableOp+block6_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block6/conv2/Conv2D/ReadVariableOpε
block6/conv2/Conv2DConv2D activation_23/Relu:activations:0*block6/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
block6/conv2/Conv2D΄
#block6/conv2/BiasAdd/ReadVariableOpReadVariableOp,block6_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block6/conv2/BiasAdd/ReadVariableOp½
block6/conv2/BiasAddBiasAddblock6/conv2/Conv2D:output:0+block6/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
block6/conv2/BiasAdd
activation_24/ReluRelublock6/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_24/Relu
activation_25/ReluRelu$block6/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
activation_25/Relu
	add_6/addAddV2 activation_24/Relu:activations:0 activation_25/Relu:activations:0*
T0*0
_output_shapes
:?????????2
	add_6/addz
activation_26/ReluReluadd_6/add:z:0*
T0*0
_output_shapes
:?????????2
activation_26/Relu€
fc1/Tensordot/ReadVariableOpReadVariableOp%fc1_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc1/Tensordot/ReadVariableOpr
fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc1/Tensordot/axes}
fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc1/Tensordot/freez
fc1/Tensordot/ShapeShape activation_26/Relu:activations:0*
T0*
_output_shapes
:2
fc1/Tensordot/Shape|
fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/GatherV2/axisε
fc1/Tensordot/GatherV2GatherV2fc1/Tensordot/Shape:output:0fc1/Tensordot/free:output:0$fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc1/Tensordot/GatherV2
fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/GatherV2_1/axisλ
fc1/Tensordot/GatherV2_1GatherV2fc1/Tensordot/Shape:output:0fc1/Tensordot/axes:output:0&fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc1/Tensordot/GatherV2_1t
fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc1/Tensordot/Const
fc1/Tensordot/ProdProdfc1/Tensordot/GatherV2:output:0fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc1/Tensordot/Prodx
fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc1/Tensordot/Const_1
fc1/Tensordot/Prod_1Prod!fc1/Tensordot/GatherV2_1:output:0fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc1/Tensordot/Prod_1x
fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/concat/axisΔ
fc1/Tensordot/concatConcatV2fc1/Tensordot/free:output:0fc1/Tensordot/axes:output:0"fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/concat
fc1/Tensordot/stackPackfc1/Tensordot/Prod:output:0fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/stack»
fc1/Tensordot/transpose	Transpose activation_26/Relu:activations:0fc1/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc1/Tensordot/transpose―
fc1/Tensordot/ReshapeReshapefc1/Tensordot/transpose:y:0fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc1/Tensordot/Reshape―
fc1/Tensordot/MatMulMatMulfc1/Tensordot/Reshape:output:0$fc1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc1/Tensordot/MatMuly
fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc1/Tensordot/Const_2|
fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc1/Tensordot/concat_1/axisΡ
fc1/Tensordot/concat_1ConcatV2fc1/Tensordot/GatherV2:output:0fc1/Tensordot/Const_2:output:0$fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc1/Tensordot/concat_1₯
fc1/TensordotReshapefc1/Tensordot/MatMul:product:0fc1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc1/Tensordot
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc1/BiasAdd/ReadVariableOp
fc1/BiasAddBiasAddfc1/Tensordot:output:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc1/BiasAddm
fc1/ReluRelufc1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc1/Relu€
fc2/Tensordot/ReadVariableOpReadVariableOp%fc2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc2/Tensordot/ReadVariableOpr
fc2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc2/Tensordot/axes}
fc2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc2/Tensordot/freep
fc2/Tensordot/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:2
fc2/Tensordot/Shape|
fc2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/GatherV2/axisε
fc2/Tensordot/GatherV2GatherV2fc2/Tensordot/Shape:output:0fc2/Tensordot/free:output:0$fc2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc2/Tensordot/GatherV2
fc2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/GatherV2_1/axisλ
fc2/Tensordot/GatherV2_1GatherV2fc2/Tensordot/Shape:output:0fc2/Tensordot/axes:output:0&fc2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc2/Tensordot/GatherV2_1t
fc2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc2/Tensordot/Const
fc2/Tensordot/ProdProdfc2/Tensordot/GatherV2:output:0fc2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc2/Tensordot/Prodx
fc2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc2/Tensordot/Const_1
fc2/Tensordot/Prod_1Prod!fc2/Tensordot/GatherV2_1:output:0fc2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc2/Tensordot/Prod_1x
fc2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/concat/axisΔ
fc2/Tensordot/concatConcatV2fc2/Tensordot/free:output:0fc2/Tensordot/axes:output:0"fc2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/concat
fc2/Tensordot/stackPackfc2/Tensordot/Prod:output:0fc2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/stack±
fc2/Tensordot/transpose	Transposefc1/Relu:activations:0fc2/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc2/Tensordot/transpose―
fc2/Tensordot/ReshapeReshapefc2/Tensordot/transpose:y:0fc2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc2/Tensordot/Reshape―
fc2/Tensordot/MatMulMatMulfc2/Tensordot/Reshape:output:0$fc2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc2/Tensordot/MatMuly
fc2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc2/Tensordot/Const_2|
fc2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc2/Tensordot/concat_1/axisΡ
fc2/Tensordot/concat_1ConcatV2fc2/Tensordot/GatherV2:output:0fc2/Tensordot/Const_2:output:0$fc2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc2/Tensordot/concat_1₯
fc2/TensordotReshapefc2/Tensordot/MatMul:product:0fc2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc2/Tensordot
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc2/BiasAdd/ReadVariableOp
fc2/BiasAddBiasAddfc2/Tensordot:output:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc2/BiasAddm
fc2/ReluRelufc2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc2/Relu€
fc3/Tensordot/ReadVariableOpReadVariableOp%fc3_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
fc3/Tensordot/ReadVariableOpr
fc3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc3/Tensordot/axes}
fc3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc3/Tensordot/freep
fc3/Tensordot/ShapeShapefc2/Relu:activations:0*
T0*
_output_shapes
:2
fc3/Tensordot/Shape|
fc3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/GatherV2/axisε
fc3/Tensordot/GatherV2GatherV2fc3/Tensordot/Shape:output:0fc3/Tensordot/free:output:0$fc3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc3/Tensordot/GatherV2
fc3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/GatherV2_1/axisλ
fc3/Tensordot/GatherV2_1GatherV2fc3/Tensordot/Shape:output:0fc3/Tensordot/axes:output:0&fc3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc3/Tensordot/GatherV2_1t
fc3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc3/Tensordot/Const
fc3/Tensordot/ProdProdfc3/Tensordot/GatherV2:output:0fc3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc3/Tensordot/Prodx
fc3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc3/Tensordot/Const_1
fc3/Tensordot/Prod_1Prod!fc3/Tensordot/GatherV2_1:output:0fc3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc3/Tensordot/Prod_1x
fc3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/concat/axisΔ
fc3/Tensordot/concatConcatV2fc3/Tensordot/free:output:0fc3/Tensordot/axes:output:0"fc3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/concat
fc3/Tensordot/stackPackfc3/Tensordot/Prod:output:0fc3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/stack±
fc3/Tensordot/transpose	Transposefc2/Relu:activations:0fc3/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc3/Tensordot/transpose―
fc3/Tensordot/ReshapeReshapefc3/Tensordot/transpose:y:0fc3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc3/Tensordot/Reshape―
fc3/Tensordot/MatMulMatMulfc3/Tensordot/Reshape:output:0$fc3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
fc3/Tensordot/MatMuly
fc3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc3/Tensordot/Const_2|
fc3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc3/Tensordot/concat_1/axisΡ
fc3/Tensordot/concat_1ConcatV2fc3/Tensordot/GatherV2:output:0fc3/Tensordot/Const_2:output:0$fc3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc3/Tensordot/concat_1₯
fc3/TensordotReshapefc3/Tensordot/MatMul:product:0fc3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
fc3/Tensordot
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
fc3/BiasAdd/ReadVariableOp
fc3/BiasAddBiasAddfc3/Tensordot:output:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
fc3/BiasAddm
fc3/ReluRelufc3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2

fc3/Relu¬
fc_out/Tensordot/ReadVariableOpReadVariableOp(fc_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02!
fc_out/Tensordot/ReadVariableOpx
fc_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
fc_out/Tensordot/axes
fc_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_out/Tensordot/freev
fc_out/Tensordot/ShapeShapefc3/Relu:activations:0*
T0*
_output_shapes
:2
fc_out/Tensordot/Shape
fc_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
fc_out/Tensordot/GatherV2/axisτ
fc_out/Tensordot/GatherV2GatherV2fc_out/Tensordot/Shape:output:0fc_out/Tensordot/free:output:0'fc_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc_out/Tensordot/GatherV2
 fc_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 fc_out/Tensordot/GatherV2_1/axisϊ
fc_out/Tensordot/GatherV2_1GatherV2fc_out/Tensordot/Shape:output:0fc_out/Tensordot/axes:output:0)fc_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
fc_out/Tensordot/GatherV2_1z
fc_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
fc_out/Tensordot/Const
fc_out/Tensordot/ProdProd"fc_out/Tensordot/GatherV2:output:0fc_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
fc_out/Tensordot/Prod~
fc_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
fc_out/Tensordot/Const_1€
fc_out/Tensordot/Prod_1Prod$fc_out/Tensordot/GatherV2_1:output:0!fc_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
fc_out/Tensordot/Prod_1~
fc_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_out/Tensordot/concat/axisΣ
fc_out/Tensordot/concatConcatV2fc_out/Tensordot/free:output:0fc_out/Tensordot/axes:output:0%fc_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/concat¨
fc_out/Tensordot/stackPackfc_out/Tensordot/Prod:output:0 fc_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/stackΊ
fc_out/Tensordot/transpose	Transposefc3/Relu:activations:0 fc_out/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
fc_out/Tensordot/transpose»
fc_out/Tensordot/ReshapeReshapefc_out/Tensordot/transpose:y:0fc_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
fc_out/Tensordot/ReshapeΊ
fc_out/Tensordot/MatMulMatMul!fc_out/Tensordot/Reshape:output:0'fc_out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fc_out/Tensordot/MatMul~
fc_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
fc_out/Tensordot/Const_2
fc_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
fc_out/Tensordot/concat_1/axisΰ
fc_out/Tensordot/concat_1ConcatV2"fc_out/Tensordot/GatherV2:output:0!fc_out/Tensordot/Const_2:output:0'fc_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
fc_out/Tensordot/concat_1°
fc_out/TensordotReshape!fc_out/Tensordot/MatMul:product:0"fc_out/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????2
fc_out/Tensordot‘
fc_out/BiasAdd/ReadVariableOpReadVariableOp&fc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_out/BiasAdd/ReadVariableOp§
fc_out/BiasAddBiasAddfc_out/Tensordot:output:0%fc_out/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
fc_out/BiasAdd~
fc_out/SigmoidSigmoidfc_out/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
fc_out/Sigmoidn
IdentityIdentityfc_out/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H:::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Φ
l
B__inference_add_3_layer_call_and_return_conditional_losses_3745231

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ
l
B__inference_add_5_layer_call_and_return_conditional_losses_3745521

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block6/conv1_layer_call_and_return_conditional_losses_3745553

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block3/conv2_layer_call_fn_3747745

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv2_layer_call_and_return_conditional_losses_37451832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
J
.__inference_activation_3_layer_call_fn_3747508

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_37448492
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ύ
z
%__inference_fc3_layer_call_fn_3748253

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc3_layer_call_and_return_conditional_losses_37458132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

«
@__inference_fc3_layer_call_and_return_conditional_losses_3748244

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block1/conv1_layer_call_fn_3747527

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv1_layer_call_and_return_conditional_losses_37448672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ρ
ί
C__inference_MTFNet_layer_call_and_return_conditional_losses_3745877
input_1
conv01_3744733
conv01_3744735
block0_conv1_3744772
block0_conv1_3744774
block0_conv2_3744811
block0_conv2_3744813
block1_conv1_3744878
block1_conv1_3744880
block1_conv2_3744917
block1_conv2_3744919
block2_conv1_3744984
block2_conv1_3744986
block2_convshortcut_3745023
block2_convshortcut_3745025
block2_conv2_3745049
block2_conv2_3745051
block3_conv1_3745129
block3_conv1_3745131
block3_convshortcut_3745168
block3_convshortcut_3745170
block3_conv2_3745194
block3_conv2_3745196
block4_conv1_3745274
block4_conv1_3745276
block4_convshortcut_3745313
block4_convshortcut_3745315
block4_conv2_3745339
block4_conv2_3745341
block5_conv1_3745419
block5_conv1_3745421
block5_convshortcut_3745458
block5_convshortcut_3745460
block5_conv2_3745484
block5_conv2_3745486
block6_conv1_3745564
block6_conv1_3745566
block6_convshortcut_3745603
block6_convshortcut_3745605
block6_conv2_3745629
block6_conv2_3745631
fc1_3745730
fc1_3745732
fc2_3745777
fc2_3745779
fc3_3745824
fc3_3745826
fc_out_3745871
fc_out_3745873
identity’$block0/conv1/StatefulPartitionedCall’$block0/conv2/StatefulPartitionedCall’$block1/conv1/StatefulPartitionedCall’$block1/conv2/StatefulPartitionedCall’$block2/conv1/StatefulPartitionedCall’$block2/conv2/StatefulPartitionedCall’+block2/convshortcut/StatefulPartitionedCall’$block3/conv1/StatefulPartitionedCall’$block3/conv2/StatefulPartitionedCall’+block3/convshortcut/StatefulPartitionedCall’$block4/conv1/StatefulPartitionedCall’$block4/conv2/StatefulPartitionedCall’+block4/convshortcut/StatefulPartitionedCall’$block5/conv1/StatefulPartitionedCall’$block5/conv2/StatefulPartitionedCall’+block5/convshortcut/StatefulPartitionedCall’$block6/conv1/StatefulPartitionedCall’$block6/conv2/StatefulPartitionedCall’+block6/convshortcut/StatefulPartitionedCall’conv01/StatefulPartitionedCall’fc1/StatefulPartitionedCall’fc2/StatefulPartitionedCall’fc3/StatefulPartitionedCall’fc_out/StatefulPartitionedCall
conv01/StatefulPartitionedCallStatefulPartitionedCallinput_1conv01_3744733conv01_3744735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv01_layer_call_and_return_conditional_losses_37447222 
conv01/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_37447432
activation/PartitionedCallΦ
$block0/conv1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0block0_conv1_3744772block0_conv1_3744774*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv1_layer_call_and_return_conditional_losses_37447612&
$block0/conv1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall-block0/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_37447822
activation_1/PartitionedCallΨ
$block0/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0block0_conv2_3744811block0_conv2_3744813*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv2_layer_call_and_return_conditional_losses_37448002&
$block0/conv2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall-block0/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_37448212
activation_2/PartitionedCall
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_37448352
add/PartitionedCall
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_37448492
activation_3/PartitionedCallΨ
$block1/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0block1_conv1_3744878block1_conv1_3744880*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv1_layer_call_and_return_conditional_losses_37448672&
$block1/conv1/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall-block1/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_37448882
activation_4/PartitionedCallΨ
$block1/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0block1_conv2_3744917block1_conv2_3744919*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv2_layer_call_and_return_conditional_losses_37449062&
$block1/conv2/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall-block1/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_37449272
activation_5/PartitionedCall‘
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_37449412
add_1/PartitionedCall
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_37449552
activation_6/PartitionedCallΨ
$block2/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_conv1_3744984block2_conv1_3744986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv1_layer_call_and_return_conditional_losses_37449732&
$block2/conv1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall-block2/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_37449942
activation_7/PartitionedCallϋ
+block2/convshortcut/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_convshortcut_3745023block2_convshortcut_3745025*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_37450122-
+block2/convshortcut/StatefulPartitionedCallΨ
$block2/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0block2_conv2_3745049block2_conv2_3745051*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv2_layer_call_and_return_conditional_losses_37450382&
$block2/conv2/StatefulPartitionedCall
activation_8/PartitionedCallPartitionedCall-block2/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_8_layer_call_and_return_conditional_losses_37450592
activation_8/PartitionedCall
activation_9/PartitionedCallPartitionedCall4block2/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_9_layer_call_and_return_conditional_losses_37450722
activation_9/PartitionedCall‘
add_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_37450862
add_2/PartitionedCall
activation_10/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_10_layer_call_and_return_conditional_losses_37451002
activation_10/PartitionedCallΩ
$block3/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_conv1_3745129block3_conv1_3745131*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv1_layer_call_and_return_conditional_losses_37451182&
$block3/conv1/StatefulPartitionedCall
activation_11/PartitionedCallPartitionedCall-block3/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_11_layer_call_and_return_conditional_losses_37451392
activation_11/PartitionedCallό
+block3/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_convshortcut_3745168block3_convshortcut_3745170*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_37451572-
+block3/convshortcut/StatefulPartitionedCallΩ
$block3/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_11/PartitionedCall:output:0block3_conv2_3745194block3_conv2_3745196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv2_layer_call_and_return_conditional_losses_37451832&
$block3/conv2/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall-block3/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_12_layer_call_and_return_conditional_losses_37452042
activation_12/PartitionedCall 
activation_13/PartitionedCallPartitionedCall4block3/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_13_layer_call_and_return_conditional_losses_37452172
activation_13/PartitionedCall£
add_3/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_37452312
add_3/PartitionedCall
activation_14/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_14_layer_call_and_return_conditional_losses_37452452
activation_14/PartitionedCallΩ
$block4/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_conv1_3745274block4_conv1_3745276*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv1_layer_call_and_return_conditional_losses_37452632&
$block4/conv1/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall-block4/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_15_layer_call_and_return_conditional_losses_37452842
activation_15/PartitionedCallό
+block4/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_convshortcut_3745313block4_convshortcut_3745315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_37453022-
+block4/convshortcut/StatefulPartitionedCallΩ
$block4/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0block4_conv2_3745339block4_conv2_3745341*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv2_layer_call_and_return_conditional_losses_37453282&
$block4/conv2/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall-block4/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_16_layer_call_and_return_conditional_losses_37453492
activation_16/PartitionedCall 
activation_17/PartitionedCallPartitionedCall4block4/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_17_layer_call_and_return_conditional_losses_37453622
activation_17/PartitionedCall£
add_4/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_4_layer_call_and_return_conditional_losses_37453762
add_4/PartitionedCall
activation_18/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_18_layer_call_and_return_conditional_losses_37453902
activation_18/PartitionedCallΩ
$block5/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_conv1_3745419block5_conv1_3745421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv1_layer_call_and_return_conditional_losses_37454082&
$block5/conv1/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-block5/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_19_layer_call_and_return_conditional_losses_37454292
activation_19/PartitionedCallό
+block5/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_convshortcut_3745458block5_convshortcut_3745460*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_37454472-
+block5/convshortcut/StatefulPartitionedCallΩ
$block5/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0block5_conv2_3745484block5_conv2_3745486*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv2_layer_call_and_return_conditional_losses_37454732&
$block5/conv2/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall-block5/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_20_layer_call_and_return_conditional_losses_37454942
activation_20/PartitionedCall 
activation_21/PartitionedCallPartitionedCall4block5/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_21_layer_call_and_return_conditional_losses_37455072
activation_21/PartitionedCall£
add_5/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_5_layer_call_and_return_conditional_losses_37455212
add_5/PartitionedCall
activation_22/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_22_layer_call_and_return_conditional_losses_37455352
activation_22/PartitionedCallΩ
$block6/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_conv1_3745564block6_conv1_3745566*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv1_layer_call_and_return_conditional_losses_37455532&
$block6/conv1/StatefulPartitionedCall
activation_23/PartitionedCallPartitionedCall-block6/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_23_layer_call_and_return_conditional_losses_37455742
activation_23/PartitionedCallό
+block6/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_convshortcut_3745603block6_convshortcut_3745605*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_37455922-
+block6/convshortcut/StatefulPartitionedCallΩ
$block6/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0block6_conv2_3745629block6_conv2_3745631*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv2_layer_call_and_return_conditional_losses_37456182&
$block6/conv2/StatefulPartitionedCall
activation_24/PartitionedCallPartitionedCall-block6/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_24_layer_call_and_return_conditional_losses_37456392
activation_24/PartitionedCall 
activation_25/PartitionedCallPartitionedCall4block6/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_25_layer_call_and_return_conditional_losses_37456522
activation_25/PartitionedCall£
add_6/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_6_layer_call_and_return_conditional_losses_37456662
add_6/PartitionedCall
activation_26/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_26_layer_call_and_return_conditional_losses_37456802
activation_26/PartitionedCall¬
fc1/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0fc1_3745730fc1_3745732*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_37457192
fc1/StatefulPartitionedCallͺ
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0fc2_3745777fc2_3745779*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_37457662
fc2/StatefulPartitionedCallͺ
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0fc3_3745824fc3_3745826*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc3_layer_call_and_return_conditional_losses_37458132
fc3/StatefulPartitionedCallΈ
fc_out/StatefulPartitionedCallStatefulPartitionedCall$fc3/StatefulPartitionedCall:output:0fc_out_3745871fc_out_3745873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_fc_out_layer_call_and_return_conditional_losses_37458602 
fc_out/StatefulPartitionedCall§
IdentityIdentity'fc_out/StatefulPartitionedCall:output:0%^block0/conv1/StatefulPartitionedCall%^block0/conv2/StatefulPartitionedCall%^block1/conv1/StatefulPartitionedCall%^block1/conv2/StatefulPartitionedCall%^block2/conv1/StatefulPartitionedCall%^block2/conv2/StatefulPartitionedCall,^block2/convshortcut/StatefulPartitionedCall%^block3/conv1/StatefulPartitionedCall%^block3/conv2/StatefulPartitionedCall,^block3/convshortcut/StatefulPartitionedCall%^block4/conv1/StatefulPartitionedCall%^block4/conv2/StatefulPartitionedCall,^block4/convshortcut/StatefulPartitionedCall%^block5/conv1/StatefulPartitionedCall%^block5/conv2/StatefulPartitionedCall,^block5/convshortcut/StatefulPartitionedCall%^block6/conv1/StatefulPartitionedCall%^block6/conv2/StatefulPartitionedCall,^block6/convshortcut/StatefulPartitionedCall^conv01/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall^fc_out/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::2L
$block0/conv1/StatefulPartitionedCall$block0/conv1/StatefulPartitionedCall2L
$block0/conv2/StatefulPartitionedCall$block0/conv2/StatefulPartitionedCall2L
$block1/conv1/StatefulPartitionedCall$block1/conv1/StatefulPartitionedCall2L
$block1/conv2/StatefulPartitionedCall$block1/conv2/StatefulPartitionedCall2L
$block2/conv1/StatefulPartitionedCall$block2/conv1/StatefulPartitionedCall2L
$block2/conv2/StatefulPartitionedCall$block2/conv2/StatefulPartitionedCall2Z
+block2/convshortcut/StatefulPartitionedCall+block2/convshortcut/StatefulPartitionedCall2L
$block3/conv1/StatefulPartitionedCall$block3/conv1/StatefulPartitionedCall2L
$block3/conv2/StatefulPartitionedCall$block3/conv2/StatefulPartitionedCall2Z
+block3/convshortcut/StatefulPartitionedCall+block3/convshortcut/StatefulPartitionedCall2L
$block4/conv1/StatefulPartitionedCall$block4/conv1/StatefulPartitionedCall2L
$block4/conv2/StatefulPartitionedCall$block4/conv2/StatefulPartitionedCall2Z
+block4/convshortcut/StatefulPartitionedCall+block4/convshortcut/StatefulPartitionedCall2L
$block5/conv1/StatefulPartitionedCall$block5/conv1/StatefulPartitionedCall2L
$block5/conv2/StatefulPartitionedCall$block5/conv2/StatefulPartitionedCall2Z
+block5/convshortcut/StatefulPartitionedCall+block5/convshortcut/StatefulPartitionedCall2L
$block6/conv1/StatefulPartitionedCall$block6/conv1/StatefulPartitionedCall2L
$block6/conv2/StatefulPartitionedCall$block6/conv2/StatefulPartitionedCall2Z
+block6/convshortcut/StatefulPartitionedCall+block6/convshortcut/StatefulPartitionedCall2@
conv01/StatefulPartitionedCallconv01/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall2@
fc_out/StatefulPartitionedCallfc_out/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
²
±
I__inference_block5/conv1_layer_call_and_return_conditional_losses_3745408

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_18_layer_call_fn_3747915

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_18_layer_call_and_return_conditional_losses_37453902
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_9_layer_call_and_return_conditional_losses_3745072

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_15_layer_call_fn_3747835

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_15_layer_call_and_return_conditional_losses_37452842
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_22_layer_call_fn_3748024

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_22_layer_call_and_return_conditional_losses_37455352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block6/conv2_layer_call_and_return_conditional_losses_3745618

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ώ
ΘT
#__inference__traced_restore_3749244
file_prefix"
assignvariableop_conv01_kernel"
assignvariableop_1_conv01_bias*
&assignvariableop_2_block0_conv1_kernel(
$assignvariableop_3_block0_conv1_bias*
&assignvariableop_4_block0_conv2_kernel(
$assignvariableop_5_block0_conv2_bias*
&assignvariableop_6_block1_conv1_kernel(
$assignvariableop_7_block1_conv1_bias*
&assignvariableop_8_block1_conv2_kernel(
$assignvariableop_9_block1_conv2_bias+
'assignvariableop_10_block2_conv1_kernel)
%assignvariableop_11_block2_conv1_bias+
'assignvariableop_12_block2_conv2_kernel)
%assignvariableop_13_block2_conv2_bias2
.assignvariableop_14_block2_convshortcut_kernel0
,assignvariableop_15_block2_convshortcut_bias+
'assignvariableop_16_block3_conv1_kernel)
%assignvariableop_17_block3_conv1_bias+
'assignvariableop_18_block3_conv2_kernel)
%assignvariableop_19_block3_conv2_bias2
.assignvariableop_20_block3_convshortcut_kernel0
,assignvariableop_21_block3_convshortcut_bias+
'assignvariableop_22_block4_conv1_kernel)
%assignvariableop_23_block4_conv1_bias+
'assignvariableop_24_block4_conv2_kernel)
%assignvariableop_25_block4_conv2_bias2
.assignvariableop_26_block4_convshortcut_kernel0
,assignvariableop_27_block4_convshortcut_bias+
'assignvariableop_28_block5_conv1_kernel)
%assignvariableop_29_block5_conv1_bias+
'assignvariableop_30_block5_conv2_kernel)
%assignvariableop_31_block5_conv2_bias2
.assignvariableop_32_block5_convshortcut_kernel0
,assignvariableop_33_block5_convshortcut_bias+
'assignvariableop_34_block6_conv1_kernel)
%assignvariableop_35_block6_conv1_bias+
'assignvariableop_36_block6_conv2_kernel)
%assignvariableop_37_block6_conv2_bias2
.assignvariableop_38_block6_convshortcut_kernel0
,assignvariableop_39_block6_convshortcut_bias"
assignvariableop_40_fc1_kernel 
assignvariableop_41_fc1_bias"
assignvariableop_42_fc2_kernel 
assignvariableop_43_fc2_bias"
assignvariableop_44_fc3_kernel 
assignvariableop_45_fc3_bias%
!assignvariableop_46_fc_out_kernel#
assignvariableop_47_fc_out_bias!
assignvariableop_48_adam_iter#
assignvariableop_49_adam_beta_1#
assignvariableop_50_adam_beta_2"
assignvariableop_51_adam_decay*
&assignvariableop_52_adam_learning_rate
assignvariableop_53_total
assignvariableop_54_count
assignvariableop_55_total_1
assignvariableop_56_count_1,
(assignvariableop_57_adam_conv01_kernel_m*
&assignvariableop_58_adam_conv01_bias_m2
.assignvariableop_59_adam_block0_conv1_kernel_m0
,assignvariableop_60_adam_block0_conv1_bias_m2
.assignvariableop_61_adam_block0_conv2_kernel_m0
,assignvariableop_62_adam_block0_conv2_bias_m2
.assignvariableop_63_adam_block1_conv1_kernel_m0
,assignvariableop_64_adam_block1_conv1_bias_m2
.assignvariableop_65_adam_block1_conv2_kernel_m0
,assignvariableop_66_adam_block1_conv2_bias_m2
.assignvariableop_67_adam_block2_conv1_kernel_m0
,assignvariableop_68_adam_block2_conv1_bias_m2
.assignvariableop_69_adam_block2_conv2_kernel_m0
,assignvariableop_70_adam_block2_conv2_bias_m9
5assignvariableop_71_adam_block2_convshortcut_kernel_m7
3assignvariableop_72_adam_block2_convshortcut_bias_m2
.assignvariableop_73_adam_block3_conv1_kernel_m0
,assignvariableop_74_adam_block3_conv1_bias_m2
.assignvariableop_75_adam_block3_conv2_kernel_m0
,assignvariableop_76_adam_block3_conv2_bias_m9
5assignvariableop_77_adam_block3_convshortcut_kernel_m7
3assignvariableop_78_adam_block3_convshortcut_bias_m2
.assignvariableop_79_adam_block4_conv1_kernel_m0
,assignvariableop_80_adam_block4_conv1_bias_m2
.assignvariableop_81_adam_block4_conv2_kernel_m0
,assignvariableop_82_adam_block4_conv2_bias_m9
5assignvariableop_83_adam_block4_convshortcut_kernel_m7
3assignvariableop_84_adam_block4_convshortcut_bias_m2
.assignvariableop_85_adam_block5_conv1_kernel_m0
,assignvariableop_86_adam_block5_conv1_bias_m2
.assignvariableop_87_adam_block5_conv2_kernel_m0
,assignvariableop_88_adam_block5_conv2_bias_m9
5assignvariableop_89_adam_block5_convshortcut_kernel_m7
3assignvariableop_90_adam_block5_convshortcut_bias_m2
.assignvariableop_91_adam_block6_conv1_kernel_m0
,assignvariableop_92_adam_block6_conv1_bias_m2
.assignvariableop_93_adam_block6_conv2_kernel_m0
,assignvariableop_94_adam_block6_conv2_bias_m9
5assignvariableop_95_adam_block6_convshortcut_kernel_m7
3assignvariableop_96_adam_block6_convshortcut_bias_m)
%assignvariableop_97_adam_fc1_kernel_m'
#assignvariableop_98_adam_fc1_bias_m)
%assignvariableop_99_adam_fc2_kernel_m(
$assignvariableop_100_adam_fc2_bias_m*
&assignvariableop_101_adam_fc3_kernel_m(
$assignvariableop_102_adam_fc3_bias_m-
)assignvariableop_103_adam_fc_out_kernel_m+
'assignvariableop_104_adam_fc_out_bias_m-
)assignvariableop_105_adam_conv01_kernel_v+
'assignvariableop_106_adam_conv01_bias_v3
/assignvariableop_107_adam_block0_conv1_kernel_v1
-assignvariableop_108_adam_block0_conv1_bias_v3
/assignvariableop_109_adam_block0_conv2_kernel_v1
-assignvariableop_110_adam_block0_conv2_bias_v3
/assignvariableop_111_adam_block1_conv1_kernel_v1
-assignvariableop_112_adam_block1_conv1_bias_v3
/assignvariableop_113_adam_block1_conv2_kernel_v1
-assignvariableop_114_adam_block1_conv2_bias_v3
/assignvariableop_115_adam_block2_conv1_kernel_v1
-assignvariableop_116_adam_block2_conv1_bias_v3
/assignvariableop_117_adam_block2_conv2_kernel_v1
-assignvariableop_118_adam_block2_conv2_bias_v:
6assignvariableop_119_adam_block2_convshortcut_kernel_v8
4assignvariableop_120_adam_block2_convshortcut_bias_v3
/assignvariableop_121_adam_block3_conv1_kernel_v1
-assignvariableop_122_adam_block3_conv1_bias_v3
/assignvariableop_123_adam_block3_conv2_kernel_v1
-assignvariableop_124_adam_block3_conv2_bias_v:
6assignvariableop_125_adam_block3_convshortcut_kernel_v8
4assignvariableop_126_adam_block3_convshortcut_bias_v3
/assignvariableop_127_adam_block4_conv1_kernel_v1
-assignvariableop_128_adam_block4_conv1_bias_v3
/assignvariableop_129_adam_block4_conv2_kernel_v1
-assignvariableop_130_adam_block4_conv2_bias_v:
6assignvariableop_131_adam_block4_convshortcut_kernel_v8
4assignvariableop_132_adam_block4_convshortcut_bias_v3
/assignvariableop_133_adam_block5_conv1_kernel_v1
-assignvariableop_134_adam_block5_conv1_bias_v3
/assignvariableop_135_adam_block5_conv2_kernel_v1
-assignvariableop_136_adam_block5_conv2_bias_v:
6assignvariableop_137_adam_block5_convshortcut_kernel_v8
4assignvariableop_138_adam_block5_convshortcut_bias_v3
/assignvariableop_139_adam_block6_conv1_kernel_v1
-assignvariableop_140_adam_block6_conv1_bias_v3
/assignvariableop_141_adam_block6_conv2_kernel_v1
-assignvariableop_142_adam_block6_conv2_bias_v:
6assignvariableop_143_adam_block6_convshortcut_kernel_v8
4assignvariableop_144_adam_block6_convshortcut_bias_v*
&assignvariableop_145_adam_fc1_kernel_v(
$assignvariableop_146_adam_fc1_bias_v*
&assignvariableop_147_adam_fc2_kernel_v(
$assignvariableop_148_adam_fc2_bias_v*
&assignvariableop_149_adam_fc3_kernel_v(
$assignvariableop_150_adam_fc3_bias_v-
)assignvariableop_151_adam_fc_out_kernel_v+
'assignvariableop_152_adam_fc_out_bias_v
identity_154’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_100’AssignVariableOp_101’AssignVariableOp_102’AssignVariableOp_103’AssignVariableOp_104’AssignVariableOp_105’AssignVariableOp_106’AssignVariableOp_107’AssignVariableOp_108’AssignVariableOp_109’AssignVariableOp_11’AssignVariableOp_110’AssignVariableOp_111’AssignVariableOp_112’AssignVariableOp_113’AssignVariableOp_114’AssignVariableOp_115’AssignVariableOp_116’AssignVariableOp_117’AssignVariableOp_118’AssignVariableOp_119’AssignVariableOp_12’AssignVariableOp_120’AssignVariableOp_121’AssignVariableOp_122’AssignVariableOp_123’AssignVariableOp_124’AssignVariableOp_125’AssignVariableOp_126’AssignVariableOp_127’AssignVariableOp_128’AssignVariableOp_129’AssignVariableOp_13’AssignVariableOp_130’AssignVariableOp_131’AssignVariableOp_132’AssignVariableOp_133’AssignVariableOp_134’AssignVariableOp_135’AssignVariableOp_136’AssignVariableOp_137’AssignVariableOp_138’AssignVariableOp_139’AssignVariableOp_14’AssignVariableOp_140’AssignVariableOp_141’AssignVariableOp_142’AssignVariableOp_143’AssignVariableOp_144’AssignVariableOp_145’AssignVariableOp_146’AssignVariableOp_147’AssignVariableOp_148’AssignVariableOp_149’AssignVariableOp_15’AssignVariableOp_150’AssignVariableOp_151’AssignVariableOp_152’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_82’AssignVariableOp_83’AssignVariableOp_84’AssignVariableOp_85’AssignVariableOp_86’AssignVariableOp_87’AssignVariableOp_88’AssignVariableOp_89’AssignVariableOp_9’AssignVariableOp_90’AssignVariableOp_91’AssignVariableOp_92’AssignVariableOp_93’AssignVariableOp_94’AssignVariableOp_95’AssignVariableOp_96’AssignVariableOp_97’AssignVariableOp_98’AssignVariableOp_99ͺX
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*΅W
value«WB¨WB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesΗ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Κ
valueΐB½B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices΄
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ώ
_output_shapesλ
θ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*«
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv01_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv01_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block0_conv1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block0_conv1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4«
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block0_conv2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5©
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block0_conv2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6«
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block1_conv1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7©
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block1_conv1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8«
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block1_conv2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block1_conv2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10―
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block2_conv1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11­
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block2_conv1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12―
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block2_conv2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13­
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block2_conv2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ά
AssignVariableOp_14AssignVariableOp.assignvariableop_14_block2_convshortcut_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15΄
AssignVariableOp_15AssignVariableOp,assignvariableop_15_block2_convshortcut_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16―
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block3_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17­
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block3_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18―
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block3_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19­
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block3_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ά
AssignVariableOp_20AssignVariableOp.assignvariableop_20_block3_convshortcut_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21΄
AssignVariableOp_21AssignVariableOp,assignvariableop_21_block3_convshortcut_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22―
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23­
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24―
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block4_conv2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25­
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block4_conv2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ά
AssignVariableOp_26AssignVariableOp.assignvariableop_26_block4_convshortcut_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27΄
AssignVariableOp_27AssignVariableOp,assignvariableop_27_block4_convshortcut_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28―
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv1_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29­
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv1_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30―
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block5_conv2_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31­
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block5_conv2_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ά
AssignVariableOp_32AssignVariableOp.assignvariableop_32_block5_convshortcut_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33΄
AssignVariableOp_33AssignVariableOp,assignvariableop_33_block5_convshortcut_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34―
AssignVariableOp_34AssignVariableOp'assignvariableop_34_block6_conv1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35­
AssignVariableOp_35AssignVariableOp%assignvariableop_35_block6_conv1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36―
AssignVariableOp_36AssignVariableOp'assignvariableop_36_block6_conv2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37­
AssignVariableOp_37AssignVariableOp%assignvariableop_37_block6_conv2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ά
AssignVariableOp_38AssignVariableOp.assignvariableop_38_block6_convshortcut_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39΄
AssignVariableOp_39AssignVariableOp,assignvariableop_39_block6_convshortcut_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¦
AssignVariableOp_40AssignVariableOpassignvariableop_40_fc1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41€
AssignVariableOp_41AssignVariableOpassignvariableop_41_fc1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¦
AssignVariableOp_42AssignVariableOpassignvariableop_42_fc2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43€
AssignVariableOp_43AssignVariableOpassignvariableop_43_fc2_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¦
AssignVariableOp_44AssignVariableOpassignvariableop_44_fc3_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45€
AssignVariableOp_45AssignVariableOpassignvariableop_45_fc3_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46©
AssignVariableOp_46AssignVariableOp!assignvariableop_46_fc_out_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47§
AssignVariableOp_47AssignVariableOpassignvariableop_47_fc_out_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_48₯
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_iterIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49§
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_beta_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50§
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_beta_2Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¦
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_decayIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_learning_rateIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53‘
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54‘
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55£
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56£
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57°
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv01_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_conv01_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ά
AssignVariableOp_59AssignVariableOp.assignvariableop_59_adam_block0_conv1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60΄
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_block0_conv1_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ά
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_block0_conv2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62΄
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_block0_conv2_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ά
AssignVariableOp_63AssignVariableOp.assignvariableop_63_adam_block1_conv1_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64΄
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_block1_conv1_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ά
AssignVariableOp_65AssignVariableOp.assignvariableop_65_adam_block1_conv2_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66΄
AssignVariableOp_66AssignVariableOp,assignvariableop_66_adam_block1_conv2_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ά
AssignVariableOp_67AssignVariableOp.assignvariableop_67_adam_block2_conv1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68΄
AssignVariableOp_68AssignVariableOp,assignvariableop_68_adam_block2_conv1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ά
AssignVariableOp_69AssignVariableOp.assignvariableop_69_adam_block2_conv2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70΄
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_block2_conv2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71½
AssignVariableOp_71AssignVariableOp5assignvariableop_71_adam_block2_convshortcut_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72»
AssignVariableOp_72AssignVariableOp3assignvariableop_72_adam_block2_convshortcut_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ά
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_block3_conv1_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74΄
AssignVariableOp_74AssignVariableOp,assignvariableop_74_adam_block3_conv1_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ά
AssignVariableOp_75AssignVariableOp.assignvariableop_75_adam_block3_conv2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76΄
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_block3_conv2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77½
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_block3_convshortcut_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78»
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adam_block3_convshortcut_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Ά
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_block4_conv1_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80΄
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_block4_conv1_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Ά
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adam_block4_conv2_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82΄
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_block4_conv2_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83½
AssignVariableOp_83AssignVariableOp5assignvariableop_83_adam_block4_convshortcut_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84»
AssignVariableOp_84AssignVariableOp3assignvariableop_84_adam_block4_convshortcut_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ά
AssignVariableOp_85AssignVariableOp.assignvariableop_85_adam_block5_conv1_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86΄
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_block5_conv1_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87Ά
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_block5_conv2_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88΄
AssignVariableOp_88AssignVariableOp,assignvariableop_88_adam_block5_conv2_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89½
AssignVariableOp_89AssignVariableOp5assignvariableop_89_adam_block5_convshortcut_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90»
AssignVariableOp_90AssignVariableOp3assignvariableop_90_adam_block5_convshortcut_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91Ά
AssignVariableOp_91AssignVariableOp.assignvariableop_91_adam_block6_conv1_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92΄
AssignVariableOp_92AssignVariableOp,assignvariableop_92_adam_block6_conv1_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Ά
AssignVariableOp_93AssignVariableOp.assignvariableop_93_adam_block6_conv2_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94΄
AssignVariableOp_94AssignVariableOp,assignvariableop_94_adam_block6_conv2_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95½
AssignVariableOp_95AssignVariableOp5assignvariableop_95_adam_block6_convshortcut_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96»
AssignVariableOp_96AssignVariableOp3assignvariableop_96_adam_block6_convshortcut_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97­
AssignVariableOp_97AssignVariableOp%assignvariableop_97_adam_fc1_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98«
AssignVariableOp_98AssignVariableOp#assignvariableop_98_adam_fc1_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99­
AssignVariableOp_99AssignVariableOp%assignvariableop_99_adam_fc2_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100―
AssignVariableOp_100AssignVariableOp$assignvariableop_100_adam_fc2_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101±
AssignVariableOp_101AssignVariableOp&assignvariableop_101_adam_fc3_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102―
AssignVariableOp_102AssignVariableOp$assignvariableop_102_adam_fc3_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103΄
AssignVariableOp_103AssignVariableOp)assignvariableop_103_adam_fc_out_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104²
AssignVariableOp_104AssignVariableOp'assignvariableop_104_adam_fc_out_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105΄
AssignVariableOp_105AssignVariableOp)assignvariableop_105_adam_conv01_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106²
AssignVariableOp_106AssignVariableOp'assignvariableop_106_adam_conv01_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107Ί
AssignVariableOp_107AssignVariableOp/assignvariableop_107_adam_block0_conv1_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108Έ
AssignVariableOp_108AssignVariableOp-assignvariableop_108_adam_block0_conv1_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109Ί
AssignVariableOp_109AssignVariableOp/assignvariableop_109_adam_block0_conv2_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110Έ
AssignVariableOp_110AssignVariableOp-assignvariableop_110_adam_block0_conv2_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111Ί
AssignVariableOp_111AssignVariableOp/assignvariableop_111_adam_block1_conv1_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112Έ
AssignVariableOp_112AssignVariableOp-assignvariableop_112_adam_block1_conv1_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113Ί
AssignVariableOp_113AssignVariableOp/assignvariableop_113_adam_block1_conv2_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114Έ
AssignVariableOp_114AssignVariableOp-assignvariableop_114_adam_block1_conv2_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115Ί
AssignVariableOp_115AssignVariableOp/assignvariableop_115_adam_block2_conv1_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116Έ
AssignVariableOp_116AssignVariableOp-assignvariableop_116_adam_block2_conv1_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117Ί
AssignVariableOp_117AssignVariableOp/assignvariableop_117_adam_block2_conv2_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118Έ
AssignVariableOp_118AssignVariableOp-assignvariableop_118_adam_block2_conv2_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119Α
AssignVariableOp_119AssignVariableOp6assignvariableop_119_adam_block2_convshortcut_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120Ώ
AssignVariableOp_120AssignVariableOp4assignvariableop_120_adam_block2_convshortcut_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121Ί
AssignVariableOp_121AssignVariableOp/assignvariableop_121_adam_block3_conv1_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122Έ
AssignVariableOp_122AssignVariableOp-assignvariableop_122_adam_block3_conv1_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123Ί
AssignVariableOp_123AssignVariableOp/assignvariableop_123_adam_block3_conv2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124Έ
AssignVariableOp_124AssignVariableOp-assignvariableop_124_adam_block3_conv2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125Α
AssignVariableOp_125AssignVariableOp6assignvariableop_125_adam_block3_convshortcut_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126Ώ
AssignVariableOp_126AssignVariableOp4assignvariableop_126_adam_block3_convshortcut_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127Ί
AssignVariableOp_127AssignVariableOp/assignvariableop_127_adam_block4_conv1_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128Έ
AssignVariableOp_128AssignVariableOp-assignvariableop_128_adam_block4_conv1_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129Ί
AssignVariableOp_129AssignVariableOp/assignvariableop_129_adam_block4_conv2_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130Έ
AssignVariableOp_130AssignVariableOp-assignvariableop_130_adam_block4_conv2_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131Α
AssignVariableOp_131AssignVariableOp6assignvariableop_131_adam_block4_convshortcut_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132Ώ
AssignVariableOp_132AssignVariableOp4assignvariableop_132_adam_block4_convshortcut_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133Ί
AssignVariableOp_133AssignVariableOp/assignvariableop_133_adam_block5_conv1_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134Έ
AssignVariableOp_134AssignVariableOp-assignvariableop_134_adam_block5_conv1_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135Ί
AssignVariableOp_135AssignVariableOp/assignvariableop_135_adam_block5_conv2_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136Έ
AssignVariableOp_136AssignVariableOp-assignvariableop_136_adam_block5_conv2_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137Α
AssignVariableOp_137AssignVariableOp6assignvariableop_137_adam_block5_convshortcut_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138Ώ
AssignVariableOp_138AssignVariableOp4assignvariableop_138_adam_block5_convshortcut_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139Ί
AssignVariableOp_139AssignVariableOp/assignvariableop_139_adam_block6_conv1_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140Έ
AssignVariableOp_140AssignVariableOp-assignvariableop_140_adam_block6_conv1_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141Ί
AssignVariableOp_141AssignVariableOp/assignvariableop_141_adam_block6_conv2_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142Έ
AssignVariableOp_142AssignVariableOp-assignvariableop_142_adam_block6_conv2_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143Α
AssignVariableOp_143AssignVariableOp6assignvariableop_143_adam_block6_convshortcut_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144Ώ
AssignVariableOp_144AssignVariableOp4assignvariableop_144_adam_block6_convshortcut_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145±
AssignVariableOp_145AssignVariableOp&assignvariableop_145_adam_fc1_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146―
AssignVariableOp_146AssignVariableOp$assignvariableop_146_adam_fc1_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147±
AssignVariableOp_147AssignVariableOp&assignvariableop_147_adam_fc2_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148―
AssignVariableOp_148AssignVariableOp$assignvariableop_148_adam_fc2_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149±
AssignVariableOp_149AssignVariableOp&assignvariableop_149_adam_fc3_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150―
AssignVariableOp_150AssignVariableOp$assignvariableop_150_adam_fc3_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151΄
AssignVariableOp_151AssignVariableOp)assignvariableop_151_adam_fc_out_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152²
AssignVariableOp_152AssignVariableOp'assignvariableop_152_adam_fc_out_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp»
Identity_153Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_153―
Identity_154IdentityIdentity_153:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_154"%
identity_154Identity_154:output:0*ϋ
_input_shapesι
ζ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ή
n
B__inference_add_6_layer_call_and_return_conditional_losses_3748117
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1


5__inference_block5/convshortcut_layer_call_fn_3747982

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_37454472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block6/conv1_layer_call_fn_3748043

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv1_layer_call_and_return_conditional_losses_37455532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ω
c
G__inference_activation_layer_call_and_return_conditional_losses_3747423

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ε
J
.__inference_activation_1_layer_call_fn_3747457

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_37447822
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
©
«
C__inference_conv01_layer_call_and_return_conditional_losses_3747409

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:H*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  H:::W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs


5__inference_block4/convshortcut_layer_call_fn_3747873

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_37453022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block3/conv2_layer_call_and_return_conditional_losses_3745183

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_12_layer_call_and_return_conditional_losses_3745204

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block2/conv2_layer_call_and_return_conditional_losses_3745038

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block4/conv2_layer_call_fn_3747854

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv2_layer_call_and_return_conditional_losses_37453282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_6_layer_call_and_return_conditional_losses_3744955

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Η
K
/__inference_activation_25_layer_call_fn_3748111

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_25_layer_call_and_return_conditional_losses_37456522
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
S
'__inference_add_6_layer_call_fn_3748123
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_6_layer_call_and_return_conditional_losses_37456662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_19_layer_call_and_return_conditional_losses_3745429

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_14_layer_call_and_return_conditional_losses_3745245

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_19_layer_call_and_return_conditional_losses_3747939

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_23_layer_call_and_return_conditional_losses_3748048

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block5/conv1_layer_call_fn_3747934

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv1_layer_call_and_return_conditional_losses_37454082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
J
.__inference_activation_5_layer_call_fn_3747566

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_37449272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ή
Έ
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_3747864

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
©
«
C__inference_conv01_layer_call_and_return_conditional_losses_3744722

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:H*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  H:::W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
α
ή
"__inference__wrapped_model_3744708
input_10
,mtfnet_conv01_conv2d_readvariableop_resource1
-mtfnet_conv01_biasadd_readvariableop_resource6
2mtfnet_block0_conv1_conv2d_readvariableop_resource7
3mtfnet_block0_conv1_biasadd_readvariableop_resource6
2mtfnet_block0_conv2_conv2d_readvariableop_resource7
3mtfnet_block0_conv2_biasadd_readvariableop_resource6
2mtfnet_block1_conv1_conv2d_readvariableop_resource7
3mtfnet_block1_conv1_biasadd_readvariableop_resource6
2mtfnet_block1_conv2_conv2d_readvariableop_resource7
3mtfnet_block1_conv2_biasadd_readvariableop_resource6
2mtfnet_block2_conv1_conv2d_readvariableop_resource7
3mtfnet_block2_conv1_biasadd_readvariableop_resource=
9mtfnet_block2_convshortcut_conv2d_readvariableop_resource>
:mtfnet_block2_convshortcut_biasadd_readvariableop_resource6
2mtfnet_block2_conv2_conv2d_readvariableop_resource7
3mtfnet_block2_conv2_biasadd_readvariableop_resource6
2mtfnet_block3_conv1_conv2d_readvariableop_resource7
3mtfnet_block3_conv1_biasadd_readvariableop_resource=
9mtfnet_block3_convshortcut_conv2d_readvariableop_resource>
:mtfnet_block3_convshortcut_biasadd_readvariableop_resource6
2mtfnet_block3_conv2_conv2d_readvariableop_resource7
3mtfnet_block3_conv2_biasadd_readvariableop_resource6
2mtfnet_block4_conv1_conv2d_readvariableop_resource7
3mtfnet_block4_conv1_biasadd_readvariableop_resource=
9mtfnet_block4_convshortcut_conv2d_readvariableop_resource>
:mtfnet_block4_convshortcut_biasadd_readvariableop_resource6
2mtfnet_block4_conv2_conv2d_readvariableop_resource7
3mtfnet_block4_conv2_biasadd_readvariableop_resource6
2mtfnet_block5_conv1_conv2d_readvariableop_resource7
3mtfnet_block5_conv1_biasadd_readvariableop_resource=
9mtfnet_block5_convshortcut_conv2d_readvariableop_resource>
:mtfnet_block5_convshortcut_biasadd_readvariableop_resource6
2mtfnet_block5_conv2_conv2d_readvariableop_resource7
3mtfnet_block5_conv2_biasadd_readvariableop_resource6
2mtfnet_block6_conv1_conv2d_readvariableop_resource7
3mtfnet_block6_conv1_biasadd_readvariableop_resource=
9mtfnet_block6_convshortcut_conv2d_readvariableop_resource>
:mtfnet_block6_convshortcut_biasadd_readvariableop_resource6
2mtfnet_block6_conv2_conv2d_readvariableop_resource7
3mtfnet_block6_conv2_biasadd_readvariableop_resource0
,mtfnet_fc1_tensordot_readvariableop_resource.
*mtfnet_fc1_biasadd_readvariableop_resource0
,mtfnet_fc2_tensordot_readvariableop_resource.
*mtfnet_fc2_biasadd_readvariableop_resource0
,mtfnet_fc3_tensordot_readvariableop_resource.
*mtfnet_fc3_biasadd_readvariableop_resource3
/mtfnet_fc_out_tensordot_readvariableop_resource1
-mtfnet_fc_out_biasadd_readvariableop_resource
identityΐ
#MTFNet/conv01/Conv2D/ReadVariableOpReadVariableOp,mtfnet_conv01_conv2d_readvariableop_resource*'
_output_shapes
:H*
dtype02%
#MTFNet/conv01/Conv2D/ReadVariableOpΟ
MTFNet/conv01/Conv2DConv2Dinput_1+MTFNet/conv01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
MTFNet/conv01/Conv2D·
$MTFNet/conv01/BiasAdd/ReadVariableOpReadVariableOp-mtfnet_conv01_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$MTFNet/conv01/BiasAdd/ReadVariableOpΑ
MTFNet/conv01/BiasAddBiasAddMTFNet/conv01/Conv2D:output:0,MTFNet/conv01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
MTFNet/conv01/BiasAdd
MTFNet/activation/ReluReluMTFNet/conv01/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation/ReluΣ
)MTFNet/block0/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block0_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block0/conv1/Conv2D/ReadVariableOpώ
MTFNet/block0/conv1/Conv2DConv2D$MTFNet/activation/Relu:activations:01MTFNet/block0/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
MTFNet/block0/conv1/Conv2DΙ
*MTFNet/block0/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block0_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block0/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block0/conv1/BiasAddBiasAdd#MTFNet/block0/conv1/Conv2D:output:02MTFNet/block0/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
MTFNet/block0/conv1/BiasAdd
MTFNet/activation_1/ReluRelu$MTFNet/block0/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_1/ReluΣ
)MTFNet/block0/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block0_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block0/conv2/Conv2D/ReadVariableOp
MTFNet/block0/conv2/Conv2DConv2D&MTFNet/activation_1/Relu:activations:01MTFNet/block0/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
MTFNet/block0/conv2/Conv2DΙ
*MTFNet/block0/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block0_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block0/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block0/conv2/BiasAddBiasAdd#MTFNet/block0/conv2/Conv2D:output:02MTFNet/block0/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
MTFNet/block0/conv2/BiasAdd
MTFNet/activation_2/ReluRelu$MTFNet/block0/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_2/Relu²
MTFNet/add/addAddV2&MTFNet/activation_2/Relu:activations:0$MTFNet/activation/Relu:activations:0*
T0*0
_output_shapes
:?????????  2
MTFNet/add/add
MTFNet/activation_3/ReluReluMTFNet/add/add:z:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_3/ReluΣ
)MTFNet/block1/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block1_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block1/conv1/Conv2D/ReadVariableOp
MTFNet/block1/conv1/Conv2DConv2D&MTFNet/activation_3/Relu:activations:01MTFNet/block1/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
MTFNet/block1/conv1/Conv2DΙ
*MTFNet/block1/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block1_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block1/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block1/conv1/BiasAddBiasAdd#MTFNet/block1/conv1/Conv2D:output:02MTFNet/block1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
MTFNet/block1/conv1/BiasAdd
MTFNet/activation_4/ReluRelu$MTFNet/block1/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_4/ReluΣ
)MTFNet/block1/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block1_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block1/conv2/Conv2D/ReadVariableOp
MTFNet/block1/conv2/Conv2DConv2D&MTFNet/activation_4/Relu:activations:01MTFNet/block1/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
MTFNet/block1/conv2/Conv2DΙ
*MTFNet/block1/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block1_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block1/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block1/conv2/BiasAddBiasAdd#MTFNet/block1/conv2/Conv2D:output:02MTFNet/block1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2
MTFNet/block1/conv2/BiasAdd
MTFNet/activation_5/ReluRelu$MTFNet/block1/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_5/ReluΈ
MTFNet/add_1/addAddV2&MTFNet/activation_5/Relu:activations:0&MTFNet/activation_3/Relu:activations:0*
T0*0
_output_shapes
:?????????  2
MTFNet/add_1/add
MTFNet/activation_6/ReluReluMTFNet/add_1/add:z:0*
T0*0
_output_shapes
:?????????  2
MTFNet/activation_6/ReluΣ
)MTFNet/block2/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block2_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block2/conv1/Conv2D/ReadVariableOp
MTFNet/block2/conv1/Conv2DConv2D&MTFNet/activation_6/Relu:activations:01MTFNet/block2/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block2/conv1/Conv2DΙ
*MTFNet/block2/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block2/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block2/conv1/BiasAddBiasAdd#MTFNet/block2/conv1/Conv2D:output:02MTFNet/block2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block2/conv1/BiasAdd
MTFNet/activation_7/ReluRelu$MTFNet/block2/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_7/Reluθ
0MTFNet/block2/convshortcut/Conv2D/ReadVariableOpReadVariableOp9mtfnet_block2_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype022
0MTFNet/block2/convshortcut/Conv2D/ReadVariableOp
!MTFNet/block2/convshortcut/Conv2DConv2D&MTFNet/activation_6/Relu:activations:08MTFNet/block2/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2#
!MTFNet/block2/convshortcut/Conv2Dή
1MTFNet/block2/convshortcut/BiasAdd/ReadVariableOpReadVariableOp:mtfnet_block2_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1MTFNet/block2/convshortcut/BiasAdd/ReadVariableOpυ
"MTFNet/block2/convshortcut/BiasAddBiasAdd*MTFNet/block2/convshortcut/Conv2D:output:09MTFNet/block2/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2$
"MTFNet/block2/convshortcut/BiasAddΣ
)MTFNet/block2/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block2/conv2/Conv2D/ReadVariableOp
MTFNet/block2/conv2/Conv2DConv2D&MTFNet/activation_7/Relu:activations:01MTFNet/block2/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block2/conv2/Conv2DΙ
*MTFNet/block2/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block2/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block2/conv2/BiasAddBiasAdd#MTFNet/block2/conv2/Conv2D:output:02MTFNet/block2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block2/conv2/BiasAdd
MTFNet/activation_8/ReluRelu$MTFNet/block2/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_8/Relu€
MTFNet/activation_9/ReluRelu+MTFNet/block2/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_9/ReluΈ
MTFNet/add_2/addAddV2&MTFNet/activation_8/Relu:activations:0&MTFNet/activation_9/Relu:activations:0*
T0*0
_output_shapes
:?????????2
MTFNet/add_2/add
MTFNet/activation_10/ReluReluMTFNet/add_2/add:z:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_10/ReluΣ
)MTFNet/block3/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block3/conv1/Conv2D/ReadVariableOp
MTFNet/block3/conv1/Conv2DConv2D'MTFNet/activation_10/Relu:activations:01MTFNet/block3/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block3/conv1/Conv2DΙ
*MTFNet/block3/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block3/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block3/conv1/BiasAddBiasAdd#MTFNet/block3/conv1/Conv2D:output:02MTFNet/block3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block3/conv1/BiasAdd
MTFNet/activation_11/ReluRelu$MTFNet/block3/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_11/Reluθ
0MTFNet/block3/convshortcut/Conv2D/ReadVariableOpReadVariableOp9mtfnet_block3_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype022
0MTFNet/block3/convshortcut/Conv2D/ReadVariableOp
!MTFNet/block3/convshortcut/Conv2DConv2D'MTFNet/activation_10/Relu:activations:08MTFNet/block3/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2#
!MTFNet/block3/convshortcut/Conv2Dή
1MTFNet/block3/convshortcut/BiasAdd/ReadVariableOpReadVariableOp:mtfnet_block3_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1MTFNet/block3/convshortcut/BiasAdd/ReadVariableOpυ
"MTFNet/block3/convshortcut/BiasAddBiasAdd*MTFNet/block3/convshortcut/Conv2D:output:09MTFNet/block3/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2$
"MTFNet/block3/convshortcut/BiasAddΣ
)MTFNet/block3/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block3/conv2/Conv2D/ReadVariableOp
MTFNet/block3/conv2/Conv2DConv2D'MTFNet/activation_11/Relu:activations:01MTFNet/block3/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block3/conv2/Conv2DΙ
*MTFNet/block3/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block3/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block3/conv2/BiasAddBiasAdd#MTFNet/block3/conv2/Conv2D:output:02MTFNet/block3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block3/conv2/BiasAdd
MTFNet/activation_12/ReluRelu$MTFNet/block3/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_12/Relu¦
MTFNet/activation_13/ReluRelu+MTFNet/block3/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_13/ReluΊ
MTFNet/add_3/addAddV2'MTFNet/activation_12/Relu:activations:0'MTFNet/activation_13/Relu:activations:0*
T0*0
_output_shapes
:?????????2
MTFNet/add_3/add
MTFNet/activation_14/ReluReluMTFNet/add_3/add:z:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_14/ReluΣ
)MTFNet/block4/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block4/conv1/Conv2D/ReadVariableOp
MTFNet/block4/conv1/Conv2DConv2D'MTFNet/activation_14/Relu:activations:01MTFNet/block4/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block4/conv1/Conv2DΙ
*MTFNet/block4/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block4/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block4/conv1/BiasAddBiasAdd#MTFNet/block4/conv1/Conv2D:output:02MTFNet/block4/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block4/conv1/BiasAdd
MTFNet/activation_15/ReluRelu$MTFNet/block4/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_15/Reluθ
0MTFNet/block4/convshortcut/Conv2D/ReadVariableOpReadVariableOp9mtfnet_block4_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype022
0MTFNet/block4/convshortcut/Conv2D/ReadVariableOp
!MTFNet/block4/convshortcut/Conv2DConv2D'MTFNet/activation_14/Relu:activations:08MTFNet/block4/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2#
!MTFNet/block4/convshortcut/Conv2Dή
1MTFNet/block4/convshortcut/BiasAdd/ReadVariableOpReadVariableOp:mtfnet_block4_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1MTFNet/block4/convshortcut/BiasAdd/ReadVariableOpυ
"MTFNet/block4/convshortcut/BiasAddBiasAdd*MTFNet/block4/convshortcut/Conv2D:output:09MTFNet/block4/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2$
"MTFNet/block4/convshortcut/BiasAddΣ
)MTFNet/block4/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block4/conv2/Conv2D/ReadVariableOp
MTFNet/block4/conv2/Conv2DConv2D'MTFNet/activation_15/Relu:activations:01MTFNet/block4/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block4/conv2/Conv2DΙ
*MTFNet/block4/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block4/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block4/conv2/BiasAddBiasAdd#MTFNet/block4/conv2/Conv2D:output:02MTFNet/block4/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block4/conv2/BiasAdd
MTFNet/activation_16/ReluRelu$MTFNet/block4/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_16/Relu¦
MTFNet/activation_17/ReluRelu+MTFNet/block4/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_17/ReluΊ
MTFNet/add_4/addAddV2'MTFNet/activation_16/Relu:activations:0'MTFNet/activation_17/Relu:activations:0*
T0*0
_output_shapes
:?????????2
MTFNet/add_4/add
MTFNet/activation_18/ReluReluMTFNet/add_4/add:z:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_18/ReluΣ
)MTFNet/block5/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block5/conv1/Conv2D/ReadVariableOp
MTFNet/block5/conv1/Conv2DConv2D'MTFNet/activation_18/Relu:activations:01MTFNet/block5/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block5/conv1/Conv2DΙ
*MTFNet/block5/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block5/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block5/conv1/BiasAddBiasAdd#MTFNet/block5/conv1/Conv2D:output:02MTFNet/block5/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block5/conv1/BiasAdd
MTFNet/activation_19/ReluRelu$MTFNet/block5/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_19/Reluθ
0MTFNet/block5/convshortcut/Conv2D/ReadVariableOpReadVariableOp9mtfnet_block5_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype022
0MTFNet/block5/convshortcut/Conv2D/ReadVariableOp
!MTFNet/block5/convshortcut/Conv2DConv2D'MTFNet/activation_18/Relu:activations:08MTFNet/block5/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2#
!MTFNet/block5/convshortcut/Conv2Dή
1MTFNet/block5/convshortcut/BiasAdd/ReadVariableOpReadVariableOp:mtfnet_block5_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1MTFNet/block5/convshortcut/BiasAdd/ReadVariableOpυ
"MTFNet/block5/convshortcut/BiasAddBiasAdd*MTFNet/block5/convshortcut/Conv2D:output:09MTFNet/block5/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2$
"MTFNet/block5/convshortcut/BiasAddΣ
)MTFNet/block5/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block5/conv2/Conv2D/ReadVariableOp
MTFNet/block5/conv2/Conv2DConv2D'MTFNet/activation_19/Relu:activations:01MTFNet/block5/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block5/conv2/Conv2DΙ
*MTFNet/block5/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block5/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block5/conv2/BiasAddBiasAdd#MTFNet/block5/conv2/Conv2D:output:02MTFNet/block5/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block5/conv2/BiasAdd
MTFNet/activation_20/ReluRelu$MTFNet/block5/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_20/Relu¦
MTFNet/activation_21/ReluRelu+MTFNet/block5/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_21/ReluΊ
MTFNet/add_5/addAddV2'MTFNet/activation_20/Relu:activations:0'MTFNet/activation_21/Relu:activations:0*
T0*0
_output_shapes
:?????????2
MTFNet/add_5/add
MTFNet/activation_22/ReluReluMTFNet/add_5/add:z:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_22/ReluΣ
)MTFNet/block6/conv1/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block6_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block6/conv1/Conv2D/ReadVariableOp
MTFNet/block6/conv1/Conv2DConv2D'MTFNet/activation_22/Relu:activations:01MTFNet/block6/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block6/conv1/Conv2DΙ
*MTFNet/block6/conv1/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block6_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block6/conv1/BiasAdd/ReadVariableOpΩ
MTFNet/block6/conv1/BiasAddBiasAdd#MTFNet/block6/conv1/Conv2D:output:02MTFNet/block6/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block6/conv1/BiasAdd
MTFNet/activation_23/ReluRelu$MTFNet/block6/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_23/Reluθ
0MTFNet/block6/convshortcut/Conv2D/ReadVariableOpReadVariableOp9mtfnet_block6_convshortcut_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype022
0MTFNet/block6/convshortcut/Conv2D/ReadVariableOp
!MTFNet/block6/convshortcut/Conv2DConv2D'MTFNet/activation_22/Relu:activations:08MTFNet/block6/convshortcut/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2#
!MTFNet/block6/convshortcut/Conv2Dή
1MTFNet/block6/convshortcut/BiasAdd/ReadVariableOpReadVariableOp:mtfnet_block6_convshortcut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1MTFNet/block6/convshortcut/BiasAdd/ReadVariableOpυ
"MTFNet/block6/convshortcut/BiasAddBiasAdd*MTFNet/block6/convshortcut/Conv2D:output:09MTFNet/block6/convshortcut/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2$
"MTFNet/block6/convshortcut/BiasAddΣ
)MTFNet/block6/conv2/Conv2D/ReadVariableOpReadVariableOp2mtfnet_block6_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)MTFNet/block6/conv2/Conv2D/ReadVariableOp
MTFNet/block6/conv2/Conv2DConv2D'MTFNet/activation_23/Relu:activations:01MTFNet/block6/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
MTFNet/block6/conv2/Conv2DΙ
*MTFNet/block6/conv2/BiasAdd/ReadVariableOpReadVariableOp3mtfnet_block6_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*MTFNet/block6/conv2/BiasAdd/ReadVariableOpΩ
MTFNet/block6/conv2/BiasAddBiasAdd#MTFNet/block6/conv2/Conv2D:output:02MTFNet/block6/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/block6/conv2/BiasAdd
MTFNet/activation_24/ReluRelu$MTFNet/block6/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_24/Relu¦
MTFNet/activation_25/ReluRelu+MTFNet/block6/convshortcut/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_25/ReluΊ
MTFNet/add_6/addAddV2'MTFNet/activation_24/Relu:activations:0'MTFNet/activation_25/Relu:activations:0*
T0*0
_output_shapes
:?????????2
MTFNet/add_6/add
MTFNet/activation_26/ReluReluMTFNet/add_6/add:z:0*
T0*0
_output_shapes
:?????????2
MTFNet/activation_26/ReluΉ
#MTFNet/fc1/Tensordot/ReadVariableOpReadVariableOp,mtfnet_fc1_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#MTFNet/fc1/Tensordot/ReadVariableOp
MTFNet/fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc1/Tensordot/axes
MTFNet/fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
MTFNet/fc1/Tensordot/free
MTFNet/fc1/Tensordot/ShapeShape'MTFNet/activation_26/Relu:activations:0*
T0*
_output_shapes
:2
MTFNet/fc1/Tensordot/Shape
"MTFNet/fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc1/Tensordot/GatherV2/axis
MTFNet/fc1/Tensordot/GatherV2GatherV2#MTFNet/fc1/Tensordot/Shape:output:0"MTFNet/fc1/Tensordot/free:output:0+MTFNet/fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
MTFNet/fc1/Tensordot/GatherV2
$MTFNet/fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$MTFNet/fc1/Tensordot/GatherV2_1/axis
MTFNet/fc1/Tensordot/GatherV2_1GatherV2#MTFNet/fc1/Tensordot/Shape:output:0"MTFNet/fc1/Tensordot/axes:output:0-MTFNet/fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
MTFNet/fc1/Tensordot/GatherV2_1
MTFNet/fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc1/Tensordot/Const¬
MTFNet/fc1/Tensordot/ProdProd&MTFNet/fc1/Tensordot/GatherV2:output:0#MTFNet/fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
MTFNet/fc1/Tensordot/Prod
MTFNet/fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc1/Tensordot/Const_1΄
MTFNet/fc1/Tensordot/Prod_1Prod(MTFNet/fc1/Tensordot/GatherV2_1:output:0%MTFNet/fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
MTFNet/fc1/Tensordot/Prod_1
 MTFNet/fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 MTFNet/fc1/Tensordot/concat/axisη
MTFNet/fc1/Tensordot/concatConcatV2"MTFNet/fc1/Tensordot/free:output:0"MTFNet/fc1/Tensordot/axes:output:0)MTFNet/fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc1/Tensordot/concatΈ
MTFNet/fc1/Tensordot/stackPack"MTFNet/fc1/Tensordot/Prod:output:0$MTFNet/fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc1/Tensordot/stackΧ
MTFNet/fc1/Tensordot/transpose	Transpose'MTFNet/activation_26/Relu:activations:0$MTFNet/fc1/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2 
MTFNet/fc1/Tensordot/transposeΛ
MTFNet/fc1/Tensordot/ReshapeReshape"MTFNet/fc1/Tensordot/transpose:y:0#MTFNet/fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
MTFNet/fc1/Tensordot/ReshapeΛ
MTFNet/fc1/Tensordot/MatMulMatMul%MTFNet/fc1/Tensordot/Reshape:output:0+MTFNet/fc1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MTFNet/fc1/Tensordot/MatMul
MTFNet/fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc1/Tensordot/Const_2
"MTFNet/fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc1/Tensordot/concat_1/axisτ
MTFNet/fc1/Tensordot/concat_1ConcatV2&MTFNet/fc1/Tensordot/GatherV2:output:0%MTFNet/fc1/Tensordot/Const_2:output:0+MTFNet/fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc1/Tensordot/concat_1Α
MTFNet/fc1/TensordotReshape%MTFNet/fc1/Tensordot/MatMul:product:0&MTFNet/fc1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc1/Tensordot?
!MTFNet/fc1/BiasAdd/ReadVariableOpReadVariableOp*mtfnet_fc1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!MTFNet/fc1/BiasAdd/ReadVariableOpΈ
MTFNet/fc1/BiasAddBiasAddMTFNet/fc1/Tensordot:output:0)MTFNet/fc1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc1/BiasAdd
MTFNet/fc1/ReluReluMTFNet/fc1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc1/ReluΉ
#MTFNet/fc2/Tensordot/ReadVariableOpReadVariableOp,mtfnet_fc2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#MTFNet/fc2/Tensordot/ReadVariableOp
MTFNet/fc2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc2/Tensordot/axes
MTFNet/fc2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
MTFNet/fc2/Tensordot/free
MTFNet/fc2/Tensordot/ShapeShapeMTFNet/fc1/Relu:activations:0*
T0*
_output_shapes
:2
MTFNet/fc2/Tensordot/Shape
"MTFNet/fc2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc2/Tensordot/GatherV2/axis
MTFNet/fc2/Tensordot/GatherV2GatherV2#MTFNet/fc2/Tensordot/Shape:output:0"MTFNet/fc2/Tensordot/free:output:0+MTFNet/fc2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
MTFNet/fc2/Tensordot/GatherV2
$MTFNet/fc2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$MTFNet/fc2/Tensordot/GatherV2_1/axis
MTFNet/fc2/Tensordot/GatherV2_1GatherV2#MTFNet/fc2/Tensordot/Shape:output:0"MTFNet/fc2/Tensordot/axes:output:0-MTFNet/fc2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
MTFNet/fc2/Tensordot/GatherV2_1
MTFNet/fc2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc2/Tensordot/Const¬
MTFNet/fc2/Tensordot/ProdProd&MTFNet/fc2/Tensordot/GatherV2:output:0#MTFNet/fc2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
MTFNet/fc2/Tensordot/Prod
MTFNet/fc2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc2/Tensordot/Const_1΄
MTFNet/fc2/Tensordot/Prod_1Prod(MTFNet/fc2/Tensordot/GatherV2_1:output:0%MTFNet/fc2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
MTFNet/fc2/Tensordot/Prod_1
 MTFNet/fc2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 MTFNet/fc2/Tensordot/concat/axisη
MTFNet/fc2/Tensordot/concatConcatV2"MTFNet/fc2/Tensordot/free:output:0"MTFNet/fc2/Tensordot/axes:output:0)MTFNet/fc2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc2/Tensordot/concatΈ
MTFNet/fc2/Tensordot/stackPack"MTFNet/fc2/Tensordot/Prod:output:0$MTFNet/fc2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc2/Tensordot/stackΝ
MTFNet/fc2/Tensordot/transpose	TransposeMTFNet/fc1/Relu:activations:0$MTFNet/fc2/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2 
MTFNet/fc2/Tensordot/transposeΛ
MTFNet/fc2/Tensordot/ReshapeReshape"MTFNet/fc2/Tensordot/transpose:y:0#MTFNet/fc2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
MTFNet/fc2/Tensordot/ReshapeΛ
MTFNet/fc2/Tensordot/MatMulMatMul%MTFNet/fc2/Tensordot/Reshape:output:0+MTFNet/fc2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MTFNet/fc2/Tensordot/MatMul
MTFNet/fc2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc2/Tensordot/Const_2
"MTFNet/fc2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc2/Tensordot/concat_1/axisτ
MTFNet/fc2/Tensordot/concat_1ConcatV2&MTFNet/fc2/Tensordot/GatherV2:output:0%MTFNet/fc2/Tensordot/Const_2:output:0+MTFNet/fc2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc2/Tensordot/concat_1Α
MTFNet/fc2/TensordotReshape%MTFNet/fc2/Tensordot/MatMul:product:0&MTFNet/fc2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc2/Tensordot?
!MTFNet/fc2/BiasAdd/ReadVariableOpReadVariableOp*mtfnet_fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!MTFNet/fc2/BiasAdd/ReadVariableOpΈ
MTFNet/fc2/BiasAddBiasAddMTFNet/fc2/Tensordot:output:0)MTFNet/fc2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc2/BiasAdd
MTFNet/fc2/ReluReluMTFNet/fc2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc2/ReluΉ
#MTFNet/fc3/Tensordot/ReadVariableOpReadVariableOp,mtfnet_fc3_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#MTFNet/fc3/Tensordot/ReadVariableOp
MTFNet/fc3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc3/Tensordot/axes
MTFNet/fc3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
MTFNet/fc3/Tensordot/free
MTFNet/fc3/Tensordot/ShapeShapeMTFNet/fc2/Relu:activations:0*
T0*
_output_shapes
:2
MTFNet/fc3/Tensordot/Shape
"MTFNet/fc3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc3/Tensordot/GatherV2/axis
MTFNet/fc3/Tensordot/GatherV2GatherV2#MTFNet/fc3/Tensordot/Shape:output:0"MTFNet/fc3/Tensordot/free:output:0+MTFNet/fc3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
MTFNet/fc3/Tensordot/GatherV2
$MTFNet/fc3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$MTFNet/fc3/Tensordot/GatherV2_1/axis
MTFNet/fc3/Tensordot/GatherV2_1GatherV2#MTFNet/fc3/Tensordot/Shape:output:0"MTFNet/fc3/Tensordot/axes:output:0-MTFNet/fc3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
MTFNet/fc3/Tensordot/GatherV2_1
MTFNet/fc3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc3/Tensordot/Const¬
MTFNet/fc3/Tensordot/ProdProd&MTFNet/fc3/Tensordot/GatherV2:output:0#MTFNet/fc3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
MTFNet/fc3/Tensordot/Prod
MTFNet/fc3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc3/Tensordot/Const_1΄
MTFNet/fc3/Tensordot/Prod_1Prod(MTFNet/fc3/Tensordot/GatherV2_1:output:0%MTFNet/fc3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
MTFNet/fc3/Tensordot/Prod_1
 MTFNet/fc3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 MTFNet/fc3/Tensordot/concat/axisη
MTFNet/fc3/Tensordot/concatConcatV2"MTFNet/fc3/Tensordot/free:output:0"MTFNet/fc3/Tensordot/axes:output:0)MTFNet/fc3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc3/Tensordot/concatΈ
MTFNet/fc3/Tensordot/stackPack"MTFNet/fc3/Tensordot/Prod:output:0$MTFNet/fc3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc3/Tensordot/stackΝ
MTFNet/fc3/Tensordot/transpose	TransposeMTFNet/fc2/Relu:activations:0$MTFNet/fc3/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2 
MTFNet/fc3/Tensordot/transposeΛ
MTFNet/fc3/Tensordot/ReshapeReshape"MTFNet/fc3/Tensordot/transpose:y:0#MTFNet/fc3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
MTFNet/fc3/Tensordot/ReshapeΛ
MTFNet/fc3/Tensordot/MatMulMatMul%MTFNet/fc3/Tensordot/Reshape:output:0+MTFNet/fc3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MTFNet/fc3/Tensordot/MatMul
MTFNet/fc3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc3/Tensordot/Const_2
"MTFNet/fc3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"MTFNet/fc3/Tensordot/concat_1/axisτ
MTFNet/fc3/Tensordot/concat_1ConcatV2&MTFNet/fc3/Tensordot/GatherV2:output:0%MTFNet/fc3/Tensordot/Const_2:output:0+MTFNet/fc3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc3/Tensordot/concat_1Α
MTFNet/fc3/TensordotReshape%MTFNet/fc3/Tensordot/MatMul:product:0&MTFNet/fc3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc3/Tensordot?
!MTFNet/fc3/BiasAdd/ReadVariableOpReadVariableOp*mtfnet_fc3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!MTFNet/fc3/BiasAdd/ReadVariableOpΈ
MTFNet/fc3/BiasAddBiasAddMTFNet/fc3/Tensordot:output:0)MTFNet/fc3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc3/BiasAdd
MTFNet/fc3/ReluReluMTFNet/fc3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
MTFNet/fc3/ReluΑ
&MTFNet/fc_out/Tensordot/ReadVariableOpReadVariableOp/mtfnet_fc_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02(
&MTFNet/fc_out/Tensordot/ReadVariableOp
MTFNet/fc_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
MTFNet/fc_out/Tensordot/axes
MTFNet/fc_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
MTFNet/fc_out/Tensordot/free
MTFNet/fc_out/Tensordot/ShapeShapeMTFNet/fc3/Relu:activations:0*
T0*
_output_shapes
:2
MTFNet/fc_out/Tensordot/Shape
%MTFNet/fc_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%MTFNet/fc_out/Tensordot/GatherV2/axis
 MTFNet/fc_out/Tensordot/GatherV2GatherV2&MTFNet/fc_out/Tensordot/Shape:output:0%MTFNet/fc_out/Tensordot/free:output:0.MTFNet/fc_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 MTFNet/fc_out/Tensordot/GatherV2
'MTFNet/fc_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'MTFNet/fc_out/Tensordot/GatherV2_1/axis
"MTFNet/fc_out/Tensordot/GatherV2_1GatherV2&MTFNet/fc_out/Tensordot/Shape:output:0%MTFNet/fc_out/Tensordot/axes:output:00MTFNet/fc_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"MTFNet/fc_out/Tensordot/GatherV2_1
MTFNet/fc_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
MTFNet/fc_out/Tensordot/ConstΈ
MTFNet/fc_out/Tensordot/ProdProd)MTFNet/fc_out/Tensordot/GatherV2:output:0&MTFNet/fc_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
MTFNet/fc_out/Tensordot/Prod
MTFNet/fc_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
MTFNet/fc_out/Tensordot/Const_1ΐ
MTFNet/fc_out/Tensordot/Prod_1Prod+MTFNet/fc_out/Tensordot/GatherV2_1:output:0(MTFNet/fc_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
MTFNet/fc_out/Tensordot/Prod_1
#MTFNet/fc_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#MTFNet/fc_out/Tensordot/concat/axisφ
MTFNet/fc_out/Tensordot/concatConcatV2%MTFNet/fc_out/Tensordot/free:output:0%MTFNet/fc_out/Tensordot/axes:output:0,MTFNet/fc_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
MTFNet/fc_out/Tensordot/concatΔ
MTFNet/fc_out/Tensordot/stackPack%MTFNet/fc_out/Tensordot/Prod:output:0'MTFNet/fc_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
MTFNet/fc_out/Tensordot/stackΦ
!MTFNet/fc_out/Tensordot/transpose	TransposeMTFNet/fc3/Relu:activations:0'MTFNet/fc_out/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????2#
!MTFNet/fc_out/Tensordot/transposeΧ
MTFNet/fc_out/Tensordot/ReshapeReshape%MTFNet/fc_out/Tensordot/transpose:y:0&MTFNet/fc_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
MTFNet/fc_out/Tensordot/ReshapeΦ
MTFNet/fc_out/Tensordot/MatMulMatMul(MTFNet/fc_out/Tensordot/Reshape:output:0.MTFNet/fc_out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
MTFNet/fc_out/Tensordot/MatMul
MTFNet/fc_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2!
MTFNet/fc_out/Tensordot/Const_2
%MTFNet/fc_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%MTFNet/fc_out/Tensordot/concat_1/axis
 MTFNet/fc_out/Tensordot/concat_1ConcatV2)MTFNet/fc_out/Tensordot/GatherV2:output:0(MTFNet/fc_out/Tensordot/Const_2:output:0.MTFNet/fc_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 MTFNet/fc_out/Tensordot/concat_1Μ
MTFNet/fc_out/TensordotReshape(MTFNet/fc_out/Tensordot/MatMul:product:0)MTFNet/fc_out/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????2
MTFNet/fc_out/TensordotΆ
$MTFNet/fc_out/BiasAdd/ReadVariableOpReadVariableOp-mtfnet_fc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$MTFNet/fc_out/BiasAdd/ReadVariableOpΓ
MTFNet/fc_out/BiasAddBiasAdd MTFNet/fc_out/Tensordot:output:0,MTFNet/fc_out/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
MTFNet/fc_out/BiasAdd
MTFNet/fc_out/SigmoidSigmoidMTFNet/fc_out/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
MTFNet/fc_out/Sigmoidu
IdentityIdentityMTFNet/fc_out/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H:::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
ύ
z
%__inference_fc2_layer_call_fn_3748213

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_37457662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ύ
z
%__inference_fc1_layer_call_fn_3748173

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_37457192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_12_layer_call_fn_3747774

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_12_layer_call_and_return_conditional_losses_37452042
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

}
(__inference_conv01_layer_call_fn_3747418

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv01_layer_call_and_return_conditional_losses_37447222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  H::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Ξ
S
'__inference_add_1_layer_call_fn_3747578
inputs_0
inputs_1
identityΫ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_37449412
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????  :?????????  :Z V
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_12_layer_call_and_return_conditional_losses_3747769

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_16_layer_call_fn_3747883

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_16_layer_call_and_return_conditional_losses_37453492
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
K
/__inference_activation_17_layer_call_fn_3747893

inputs
identityΦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_17_layer_call_and_return_conditional_losses_37453622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
n
B__inference_add_3_layer_call_and_return_conditional_losses_3747790
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ά
f
J__inference_activation_25_layer_call_and_return_conditional_losses_3745652

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3747452

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs


.__inference_block3/conv1_layer_call_fn_3747716

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv1_layer_call_and_return_conditional_losses_37451182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
Έ
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_3745012

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_24_layer_call_and_return_conditional_losses_3748096

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block1/conv1_layer_call_and_return_conditional_losses_3747518

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_13_layer_call_and_return_conditional_losses_3747779

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block0/conv1_layer_call_and_return_conditional_losses_3747438

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs

?
C__inference_fc_out_layer_call_and_return_conditional_losses_3745860

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

Υ
(__inference_MTFNet_layer_call_fn_3746295
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_MTFNet_layer_call_and_return_conditional_losses_37461962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  H
!
_user_specified_name	input_1
²
±
I__inference_block1/conv2_layer_call_and_return_conditional_losses_3747547

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  :::X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_15_layer_call_and_return_conditional_losses_3747830

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
J
.__inference_activation_8_layer_call_fn_3747665

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_8_layer_call_and_return_conditional_losses_37450592
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3744782

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ή
Έ
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_3745302

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
J
.__inference_activation_2_layer_call_fn_3747486

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_37448212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_14_layer_call_and_return_conditional_losses_3747801

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


.__inference_block2/conv1_layer_call_fn_3747607

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv1_layer_call_and_return_conditional_losses_37449732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ά
f
J__inference_activation_16_layer_call_and_return_conditional_losses_3747878

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ά
f
J__inference_activation_26_layer_call_and_return_conditional_losses_3748128

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
±
I__inference_block3/conv1_layer_call_and_return_conditional_losses_3745118

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_3_layer_call_and_return_conditional_losses_3744849

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Ϋ
e
I__inference_activation_5_layer_call_and_return_conditional_losses_3744927

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????  :X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
ρ
ή
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746455

inputs
conv01_3746300
conv01_3746302
block0_conv1_3746306
block0_conv1_3746308
block0_conv2_3746312
block0_conv2_3746314
block1_conv1_3746320
block1_conv1_3746322
block1_conv2_3746326
block1_conv2_3746328
block2_conv1_3746334
block2_conv1_3746336
block2_convshortcut_3746340
block2_convshortcut_3746342
block2_conv2_3746345
block2_conv2_3746347
block3_conv1_3746354
block3_conv1_3746356
block3_convshortcut_3746360
block3_convshortcut_3746362
block3_conv2_3746365
block3_conv2_3746367
block4_conv1_3746374
block4_conv1_3746376
block4_convshortcut_3746380
block4_convshortcut_3746382
block4_conv2_3746385
block4_conv2_3746387
block5_conv1_3746394
block5_conv1_3746396
block5_convshortcut_3746400
block5_convshortcut_3746402
block5_conv2_3746405
block5_conv2_3746407
block6_conv1_3746414
block6_conv1_3746416
block6_convshortcut_3746420
block6_convshortcut_3746422
block6_conv2_3746425
block6_conv2_3746427
fc1_3746434
fc1_3746436
fc2_3746439
fc2_3746441
fc3_3746444
fc3_3746446
fc_out_3746449
fc_out_3746451
identity’$block0/conv1/StatefulPartitionedCall’$block0/conv2/StatefulPartitionedCall’$block1/conv1/StatefulPartitionedCall’$block1/conv2/StatefulPartitionedCall’$block2/conv1/StatefulPartitionedCall’$block2/conv2/StatefulPartitionedCall’+block2/convshortcut/StatefulPartitionedCall’$block3/conv1/StatefulPartitionedCall’$block3/conv2/StatefulPartitionedCall’+block3/convshortcut/StatefulPartitionedCall’$block4/conv1/StatefulPartitionedCall’$block4/conv2/StatefulPartitionedCall’+block4/convshortcut/StatefulPartitionedCall’$block5/conv1/StatefulPartitionedCall’$block5/conv2/StatefulPartitionedCall’+block5/convshortcut/StatefulPartitionedCall’$block6/conv1/StatefulPartitionedCall’$block6/conv2/StatefulPartitionedCall’+block6/convshortcut/StatefulPartitionedCall’conv01/StatefulPartitionedCall’fc1/StatefulPartitionedCall’fc2/StatefulPartitionedCall’fc3/StatefulPartitionedCall’fc_out/StatefulPartitionedCall
conv01/StatefulPartitionedCallStatefulPartitionedCallinputsconv01_3746300conv01_3746302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_conv01_layer_call_and_return_conditional_losses_37447222 
conv01/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_37447432
activation/PartitionedCallΦ
$block0/conv1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0block0_conv1_3746306block0_conv1_3746308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv1_layer_call_and_return_conditional_losses_37447612&
$block0/conv1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall-block0/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_37447822
activation_1/PartitionedCallΨ
$block0/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0block0_conv2_3746312block0_conv2_3746314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block0/conv2_layer_call_and_return_conditional_losses_37448002&
$block0/conv2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall-block0/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_37448212
activation_2/PartitionedCall
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_37448352
add/PartitionedCall
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_37448492
activation_3/PartitionedCallΨ
$block1/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0block1_conv1_3746320block1_conv1_3746322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv1_layer_call_and_return_conditional_losses_37448672&
$block1/conv1/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall-block1/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_37448882
activation_4/PartitionedCallΨ
$block1/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0block1_conv2_3746326block1_conv2_3746328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block1/conv2_layer_call_and_return_conditional_losses_37449062&
$block1/conv2/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall-block1/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_37449272
activation_5/PartitionedCall‘
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_37449412
add_1/PartitionedCall
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_37449552
activation_6/PartitionedCallΨ
$block2/conv1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_conv1_3746334block2_conv1_3746336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv1_layer_call_and_return_conditional_losses_37449732&
$block2/conv1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall-block2/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_37449942
activation_7/PartitionedCallϋ
+block2/convshortcut/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0block2_convshortcut_3746340block2_convshortcut_3746342*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_37450122-
+block2/convshortcut/StatefulPartitionedCallΨ
$block2/conv2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0block2_conv2_3746345block2_conv2_3746347*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block2/conv2_layer_call_and_return_conditional_losses_37450382&
$block2/conv2/StatefulPartitionedCall
activation_8/PartitionedCallPartitionedCall-block2/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_8_layer_call_and_return_conditional_losses_37450592
activation_8/PartitionedCall
activation_9/PartitionedCallPartitionedCall4block2/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_9_layer_call_and_return_conditional_losses_37450722
activation_9/PartitionedCall‘
add_2/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_37450862
add_2/PartitionedCall
activation_10/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_10_layer_call_and_return_conditional_losses_37451002
activation_10/PartitionedCallΩ
$block3/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_conv1_3746354block3_conv1_3746356*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv1_layer_call_and_return_conditional_losses_37451182&
$block3/conv1/StatefulPartitionedCall
activation_11/PartitionedCallPartitionedCall-block3/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_11_layer_call_and_return_conditional_losses_37451392
activation_11/PartitionedCallό
+block3/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0block3_convshortcut_3746360block3_convshortcut_3746362*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_37451572-
+block3/convshortcut/StatefulPartitionedCallΩ
$block3/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_11/PartitionedCall:output:0block3_conv2_3746365block3_conv2_3746367*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block3/conv2_layer_call_and_return_conditional_losses_37451832&
$block3/conv2/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall-block3/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_12_layer_call_and_return_conditional_losses_37452042
activation_12/PartitionedCall 
activation_13/PartitionedCallPartitionedCall4block3/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_13_layer_call_and_return_conditional_losses_37452172
activation_13/PartitionedCall£
add_3/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_37452312
add_3/PartitionedCall
activation_14/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_14_layer_call_and_return_conditional_losses_37452452
activation_14/PartitionedCallΩ
$block4/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_conv1_3746374block4_conv1_3746376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv1_layer_call_and_return_conditional_losses_37452632&
$block4/conv1/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall-block4/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_15_layer_call_and_return_conditional_losses_37452842
activation_15/PartitionedCallό
+block4/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0block4_convshortcut_3746380block4_convshortcut_3746382*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_37453022-
+block4/convshortcut/StatefulPartitionedCallΩ
$block4/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0block4_conv2_3746385block4_conv2_3746387*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block4/conv2_layer_call_and_return_conditional_losses_37453282&
$block4/conv2/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall-block4/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_16_layer_call_and_return_conditional_losses_37453492
activation_16/PartitionedCall 
activation_17/PartitionedCallPartitionedCall4block4/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_17_layer_call_and_return_conditional_losses_37453622
activation_17/PartitionedCall£
add_4/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_4_layer_call_and_return_conditional_losses_37453762
add_4/PartitionedCall
activation_18/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_18_layer_call_and_return_conditional_losses_37453902
activation_18/PartitionedCallΩ
$block5/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_conv1_3746394block5_conv1_3746396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv1_layer_call_and_return_conditional_losses_37454082&
$block5/conv1/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-block5/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_19_layer_call_and_return_conditional_losses_37454292
activation_19/PartitionedCallό
+block5/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0block5_convshortcut_3746400block5_convshortcut_3746402*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_37454472-
+block5/convshortcut/StatefulPartitionedCallΩ
$block5/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0block5_conv2_3746405block5_conv2_3746407*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block5/conv2_layer_call_and_return_conditional_losses_37454732&
$block5/conv2/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall-block5/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_20_layer_call_and_return_conditional_losses_37454942
activation_20/PartitionedCall 
activation_21/PartitionedCallPartitionedCall4block5/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_21_layer_call_and_return_conditional_losses_37455072
activation_21/PartitionedCall£
add_5/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_5_layer_call_and_return_conditional_losses_37455212
add_5/PartitionedCall
activation_22/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_22_layer_call_and_return_conditional_losses_37455352
activation_22/PartitionedCallΩ
$block6/conv1/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_conv1_3746414block6_conv1_3746416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv1_layer_call_and_return_conditional_losses_37455532&
$block6/conv1/StatefulPartitionedCall
activation_23/PartitionedCallPartitionedCall-block6/conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_23_layer_call_and_return_conditional_losses_37455742
activation_23/PartitionedCallό
+block6/convshortcut/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0block6_convshortcut_3746420block6_convshortcut_3746422*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_37455922-
+block6/convshortcut/StatefulPartitionedCallΩ
$block6/conv2/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0block6_conv2_3746425block6_conv2_3746427*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_block6/conv2_layer_call_and_return_conditional_losses_37456182&
$block6/conv2/StatefulPartitionedCall
activation_24/PartitionedCallPartitionedCall-block6/conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_24_layer_call_and_return_conditional_losses_37456392
activation_24/PartitionedCall 
activation_25/PartitionedCallPartitionedCall4block6/convshortcut/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_25_layer_call_and_return_conditional_losses_37456522
activation_25/PartitionedCall£
add_6/PartitionedCallPartitionedCall&activation_24/PartitionedCall:output:0&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_add_6_layer_call_and_return_conditional_losses_37456662
add_6/PartitionedCall
activation_26/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_activation_26_layer_call_and_return_conditional_losses_37456802
activation_26/PartitionedCall¬
fc1/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0fc1_3746434fc1_3746436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_37457192
fc1/StatefulPartitionedCallͺ
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0fc2_3746439fc2_3746441*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_37457662
fc2/StatefulPartitionedCallͺ
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0fc3_3746444fc3_3746446*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_fc3_layer_call_and_return_conditional_losses_37458132
fc3/StatefulPartitionedCallΈ
fc_out/StatefulPartitionedCallStatefulPartitionedCall$fc3/StatefulPartitionedCall:output:0fc_out_3746449fc_out_3746451*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_fc_out_layer_call_and_return_conditional_losses_37458602 
fc_out/StatefulPartitionedCall§
IdentityIdentity'fc_out/StatefulPartitionedCall:output:0%^block0/conv1/StatefulPartitionedCall%^block0/conv2/StatefulPartitionedCall%^block1/conv1/StatefulPartitionedCall%^block1/conv2/StatefulPartitionedCall%^block2/conv1/StatefulPartitionedCall%^block2/conv2/StatefulPartitionedCall,^block2/convshortcut/StatefulPartitionedCall%^block3/conv1/StatefulPartitionedCall%^block3/conv2/StatefulPartitionedCall,^block3/convshortcut/StatefulPartitionedCall%^block4/conv1/StatefulPartitionedCall%^block4/conv2/StatefulPartitionedCall,^block4/convshortcut/StatefulPartitionedCall%^block5/conv1/StatefulPartitionedCall%^block5/conv2/StatefulPartitionedCall,^block5/convshortcut/StatefulPartitionedCall%^block6/conv1/StatefulPartitionedCall%^block6/conv2/StatefulPartitionedCall,^block6/convshortcut/StatefulPartitionedCall^conv01/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall^fc_out/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*π
_input_shapesή
Ϋ:?????????  H::::::::::::::::::::::::::::::::::::::::::::::::2L
$block0/conv1/StatefulPartitionedCall$block0/conv1/StatefulPartitionedCall2L
$block0/conv2/StatefulPartitionedCall$block0/conv2/StatefulPartitionedCall2L
$block1/conv1/StatefulPartitionedCall$block1/conv1/StatefulPartitionedCall2L
$block1/conv2/StatefulPartitionedCall$block1/conv2/StatefulPartitionedCall2L
$block2/conv1/StatefulPartitionedCall$block2/conv1/StatefulPartitionedCall2L
$block2/conv2/StatefulPartitionedCall$block2/conv2/StatefulPartitionedCall2Z
+block2/convshortcut/StatefulPartitionedCall+block2/convshortcut/StatefulPartitionedCall2L
$block3/conv1/StatefulPartitionedCall$block3/conv1/StatefulPartitionedCall2L
$block3/conv2/StatefulPartitionedCall$block3/conv2/StatefulPartitionedCall2Z
+block3/convshortcut/StatefulPartitionedCall+block3/convshortcut/StatefulPartitionedCall2L
$block4/conv1/StatefulPartitionedCall$block4/conv1/StatefulPartitionedCall2L
$block4/conv2/StatefulPartitionedCall$block4/conv2/StatefulPartitionedCall2Z
+block4/convshortcut/StatefulPartitionedCall+block4/convshortcut/StatefulPartitionedCall2L
$block5/conv1/StatefulPartitionedCall$block5/conv1/StatefulPartitionedCall2L
$block5/conv2/StatefulPartitionedCall$block5/conv2/StatefulPartitionedCall2Z
+block5/convshortcut/StatefulPartitionedCall+block5/convshortcut/StatefulPartitionedCall2L
$block6/conv1/StatefulPartitionedCall$block6/conv1/StatefulPartitionedCall2L
$block6/conv2/StatefulPartitionedCall$block6/conv2/StatefulPartitionedCall2Z
+block6/convshortcut/StatefulPartitionedCall+block6/convshortcut/StatefulPartitionedCall2@
conv01/StatefulPartitionedCallconv01/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall2@
fc_out/StatefulPartitionedCallfc_out/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  H
 
_user_specified_nameinputs
Ε
J
.__inference_activation_9_layer_call_fn_3747675

inputs
identityΥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_activation_9_layer_call_and_return_conditional_losses_37450722
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ή
serving_default₯
C
input_18
serving_default_input_1:0?????????  HB
fc_out8
StatefulPartitionedCall:0?????????tensorflow/serving/predict:αθ
²
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-8
layer-23
layer-24
layer_with_weights-9
layer-25
layer_with_weights-10
layer-26
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-11
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer_with_weights-13
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-14
(layer-39
)layer-40
*layer_with_weights-15
*layer-41
+layer_with_weights-16
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer_with_weights-17
0layer-47
1layer-48
2layer_with_weights-18
2layer-49
3layer_with_weights-19
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer_with_weights-20
8layer-55
9layer_with_weights-21
9layer-56
:layer_with_weights-22
:layer-57
;layer_with_weights-23
;layer-58
<	optimizer
=trainable_variables
>regularization_losses
?	variables
@	keras_api
A
signatures
ρ__call__
+ς&call_and_return_all_conditional_losses
σ_default_save_signature"΅ώ
_tf_keras_networkώ{"class_name": "Functional", "name": "MTFNet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "MTFNet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv01", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv01", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["conv01", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block0/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block0/conv1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["block0/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block0/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block0/conv2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["block0/conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["activation_2", 0, 0, {}], ["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1/conv1", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["block1/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1/conv2", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["block1/conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_5", 0, 0, {}], ["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/conv1", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["block2/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/conv2", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/convshortcut", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["block2/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["block2/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["activation_8", 0, 0, {}], ["activation_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/conv1", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["block3/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/conv2", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/convshortcut", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["block3/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["block3/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["activation_12", 0, 0, {}], ["activation_13", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/conv1", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["block4/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/conv2", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/convshortcut", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["block4/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["block4/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["activation_16", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/conv1", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["block5/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/conv2", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/convshortcut", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["block5/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["block5/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["activation_20", 0, 0, {}], ["activation_21", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/conv1", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["block6/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/conv2", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/convshortcut", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_24", "inbound_nodes": [[["block6/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_25", "inbound_nodes": [[["block6/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["activation_24", 0, 0, {}], ["activation_25", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_26", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["activation_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_out", "trainable": true, "dtype": "float32", "units": 8, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_out", "inbound_nodes": [[["fc3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc_out", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 72]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "MTFNet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv01", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv01", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["conv01", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block0/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block0/conv1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["block0/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block0/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block0/conv2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["block0/conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["activation_2", 0, 0, {}], ["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1/conv1", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["block1/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1/conv2", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["block1/conv2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_5", 0, 0, {}], ["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/conv1", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["block2/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/conv2", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2/convshortcut", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["block2/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["block2/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["activation_8", 0, 0, {}], ["activation_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/conv1", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["block3/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/conv2", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3/convshortcut", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["block3/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["block3/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["activation_12", 0, 0, {}], ["activation_13", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/conv1", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["block4/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/conv2", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4/convshortcut", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["block4/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["block4/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["activation_16", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/conv1", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["block5/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/conv2", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5/convshortcut", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["block5/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["block5/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["activation_20", 0, 0, {}], ["activation_21", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/conv1", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["block6/conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/conv2", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block6/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block6/convshortcut", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_24", "inbound_nodes": [[["block6/conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_25", "inbound_nodes": [[["block6/convshortcut", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["activation_24", 0, 0, {}], ["activation_25", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_26", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["activation_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_out", "trainable": true, "dtype": "float32", "units": 8, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_out", "inbound_nodes": [[["fc3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc_out", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ϋ"ψ
_tf_keras_input_layerΨ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 72]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
σ	

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
τ__call__
+υ&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv01", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv01", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 72]}}
Σ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
φ__call__
+χ&call_and_return_all_conditional_losses"Β
_tf_keras_layer¨{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}



Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
ψ__call__
+ω&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block0/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block0/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Χ
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
ϊ__call__
+ϋ&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}



Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
ό__call__
+ύ&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block0/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block0/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Χ
\	variables
]trainable_variables
^regularization_losses
_	keras_api
ώ__call__
+?&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ή
`	variables
atrainable_variables
bregularization_losses
c	keras_api
__call__
+&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 128]}, {"class_name": "TensorShape", "items": [null, 32, 32, 128]}]}
Χ
d	variables
etrainable_variables
fregularization_losses
g	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}



hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block1/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1/conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Χ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}



rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block1/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1/conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Χ
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
½
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"¬
_tf_keras_layer{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 128]}, {"class_name": "TensorShape", "items": [null, 32, 32, 128]}]}
Ϋ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block2/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Ϋ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block2/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"θ
_tf_keras_layerΞ{"class_name": "Conv2D", "name": "block2/convshortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Ϋ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ϋ
	variables
trainable_variables
 regularization_losses
‘	keras_api
__call__
+&call_and_return_all_conditional_losses"Ζ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}}
Α
’	variables
£trainable_variables
€regularization_losses
₯	keras_api
__call__
+&call_and_return_all_conditional_losses"¬
_tf_keras_layer{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 256]}, {"class_name": "TensorShape", "items": [null, 16, 16, 256]}]}
έ
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
__call__
+&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}


ͺkernel
	«bias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
 __call__
+‘&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Conv2D", "name": "block3/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
έ
°	variables
±trainable_variables
²regularization_losses
³	keras_api
’__call__
+£&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}


΄kernel
	΅bias
Ά	variables
·trainable_variables
Έregularization_losses
Ή	keras_api
€__call__
+₯&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block3/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}


Ίkernel
	»bias
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"θ
_tf_keras_layerΞ{"class_name": "Conv2D", "name": "block3/convshortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
έ
ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}
έ
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
ͺ__call__
+«&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}
½
Θ	variables
Ιtrainable_variables
Κregularization_losses
Λ	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 8, 8, 256]}, {"class_name": "TensorShape", "items": [null, 8, 8, 256]}]}
έ
Μ	variables
Νtrainable_variables
Ξregularization_losses
Ο	keras_api
?__call__
+―&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}


Πkernel
	Ρbias
?	variables
Σtrainable_variables
Τregularization_losses
Υ	keras_api
°__call__
+±&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block4/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
έ
Φ	variables
Χtrainable_variables
Ψregularization_losses
Ω	keras_api
²__call__
+³&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}


Ϊkernel
	Ϋbias
ά	variables
έtrainable_variables
ήregularization_losses
ί	keras_api
΄__call__
+΅&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block4/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 256]}}


ΰkernel
	αbias
β	variables
γtrainable_variables
δregularization_losses
ε	keras_api
Ά__call__
+·&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "Conv2D", "name": "block4/convshortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
έ
ζ	variables
ηtrainable_variables
θregularization_losses
ι	keras_api
Έ__call__
+Ή&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}
έ
κ	variables
λtrainable_variables
μregularization_losses
ν	keras_api
Ί__call__
+»&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}
½
ξ	variables
οtrainable_variables
πregularization_losses
ρ	keras_api
Ό__call__
+½&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4, 4, 256]}, {"class_name": "TensorShape", "items": [null, 4, 4, 256]}]}
έ
ς	variables
σtrainable_variables
τregularization_losses
υ	keras_api
Ύ__call__
+Ώ&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}


φkernel
	χbias
ψ	variables
ωtrainable_variables
ϊregularization_losses
ϋ	keras_api
ΐ__call__
+Α&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block5/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 256]}}
έ
ό	variables
ύtrainable_variables
ώregularization_losses
?	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
Δ__call__
+Ε&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block5/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 256]}}


kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
Ζ__call__
+Η&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "Conv2D", "name": "block5/convshortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 256]}}
έ
	variables
trainable_variables
regularization_losses
	keras_api
Θ__call__
+Ι&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
έ
	variables
trainable_variables
regularization_losses
	keras_api
Κ__call__
+Λ&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}
½
	variables
trainable_variables
regularization_losses
	keras_api
Μ__call__
+Ν&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2, 2, 256]}, {"class_name": "TensorShape", "items": [null, 2, 2, 256]}]}
έ
	variables
trainable_variables
regularization_losses
	keras_api
Ξ__call__
+Ο&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}}


kernel
	bias
	variables
trainable_variables
 regularization_losses
‘	keras_api
Π__call__
+Ρ&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block6/conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block6/conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 256]}}
έ
’	variables
£trainable_variables
€regularization_losses
₯	keras_api
?__call__
+Σ&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}}


¦kernel
	§bias
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
Τ__call__
+Υ&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Conv2D", "name": "block6/conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block6/conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 256]}}


¬kernel
	­bias
?	variables
―trainable_variables
°regularization_losses
±	keras_api
Φ__call__
+Χ&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "Conv2D", "name": "block6/convshortcut", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block6/convshortcut", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 256]}}
έ
²	variables
³trainable_variables
΄regularization_losses
΅	keras_api
Ψ__call__
+Ω&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_24", "trainable": true, "dtype": "float32", "activation": "relu"}}
έ
Ά	variables
·trainable_variables
Έregularization_losses
Ή	keras_api
Ϊ__call__
+Ϋ&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}}
½
Ί	variables
»trainable_variables
Όregularization_losses
½	keras_api
ά__call__
+έ&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Add", "name": "add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 1, 256]}, {"class_name": "TensorShape", "items": [null, 1, 1, 256]}]}
έ
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
ή__call__
+ί&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Activation", "name": "activation_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}
ω
Βkernel
	Γbias
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
ΰ__call__
+α&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 256]}}
ω
Θkernel
	Ιbias
Κ	variables
Λtrainable_variables
Μregularization_losses
Ν	keras_api
β__call__
+γ&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Dense", "name": "fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 256]}}
ω
Ξkernel
	Οbias
Π	variables
Ρtrainable_variables
?regularization_losses
Σ	keras_api
δ__call__
+ε&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Dense", "name": "fc3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 256]}}

Τkernel
	Υbias
Φ	variables
Χtrainable_variables
Ψregularization_losses
Ω	keras_api
ζ__call__
+η&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "Dense", "name": "fc_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_out", "trainable": true, "dtype": "float32", "units": 8, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 128]}}
δ
	Ϊiter
Ϋbeta_1
άbeta_2

έdecay
ήlearning_rateBmCmLmMmVmWmhmimrmsm	m	m	m	m	m	m 	ͺm‘	«m’	΄m£	΅m€	Ίm₯	»m¦	Πm§	Ρm¨	Ϊm©	Ϋmͺ	ΰm«	αm¬	φm­	χm?	m―	m°	m±	m²	m³	m΄	¦m΅	§mΆ	¬m·	­mΈ	ΒmΉ	ΓmΊ	Θm»	ΙmΌ	Ξm½	ΟmΎ	ΤmΏ	ΥmΐBvΑCvΒLvΓMvΔVvΕWvΖhvΗivΘrvΙsvΚ	vΛ	vΜ	vΝ	vΞ	vΟ	vΠ	ͺvΡ	«v?	΄vΣ	΅vΤ	ΊvΥ	»vΦ	ΠvΧ	ΡvΨ	ΪvΩ	ΫvΪ	ΰvΫ	αvά	φvέ	χvή	vί	vΰ	vα	vβ	vγ	vδ	¦vε	§vζ	¬vη	­vθ	Βvι	Γvκ	Θvλ	Ιvμ	Ξvν	Οvξ	Τvο	Υvπ"
	optimizer
Ό
B0
C1
L2
M3
V4
W5
h6
i7
r8
s9
10
11
12
13
14
15
ͺ16
«17
΄18
΅19
Ί20
»21
Π22
Ρ23
Ϊ24
Ϋ25
ΰ26
α27
φ28
χ29
30
31
32
33
34
35
¦36
§37
¬38
­39
Β40
Γ41
Θ42
Ι43
Ξ44
Ο45
Τ46
Υ47"
trackable_list_wrapper
 "
trackable_list_wrapper
Ό
B0
C1
L2
M3
V4
W5
h6
i7
r8
s9
10
11
12
13
14
15
ͺ16
«17
΄18
΅19
Ί20
»21
Π22
Ρ23
Ϊ24
Ϋ25
ΰ26
α27
φ28
χ29
30
31
32
33
34
35
¦36
§37
¬38
­39
Β40
Γ41
Θ42
Ι43
Ξ44
Ο45
Τ46
Υ47"
trackable_list_wrapper
Σ
=trainable_variables
ίlayer_metrics
ΰmetrics
αlayers
 βlayer_regularization_losses
>regularization_losses
γnon_trainable_variables
?	variables
ρ__call__
σ_default_save_signature
+ς&call_and_return_all_conditional_losses
'ς"call_and_return_conditional_losses"
_generic_user_object
-
θserving_default"
signature_map
(:&H2conv01/kernel
:2conv01/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
D	variables
Etrainable_variables
δlayer_metrics
εmetrics
ζlayers
 ηlayer_regularization_losses
Fregularization_losses
θnon_trainable_variables
τ__call__
+υ&call_and_return_all_conditional_losses
'υ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
H	variables
Itrainable_variables
ιlayer_metrics
κmetrics
λlayers
 μlayer_regularization_losses
Jregularization_losses
νnon_trainable_variables
φ__call__
+χ&call_and_return_all_conditional_losses
'χ"call_and_return_conditional_losses"
_generic_user_object
/:-2block0/conv1/kernel
 :2block0/conv1/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
N	variables
Otrainable_variables
ξlayer_metrics
οmetrics
πlayers
 ρlayer_regularization_losses
Pregularization_losses
ςnon_trainable_variables
ψ__call__
+ω&call_and_return_all_conditional_losses
'ω"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
R	variables
Strainable_variables
σlayer_metrics
τmetrics
υlayers
 φlayer_regularization_losses
Tregularization_losses
χnon_trainable_variables
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
'ϋ"call_and_return_conditional_losses"
_generic_user_object
/:-2block0/conv2/kernel
 :2block0/conv2/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
X	variables
Ytrainable_variables
ψlayer_metrics
ωmetrics
ϊlayers
 ϋlayer_regularization_losses
Zregularization_losses
όnon_trainable_variables
ό__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
\	variables
]trainable_variables
ύlayer_metrics
ώmetrics
?layers
 layer_regularization_losses
^regularization_losses
non_trainable_variables
ώ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
`	variables
atrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
bregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
d	variables
etrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
fregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-2block1/conv1/kernel
 :2block1/conv1/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
j	variables
ktrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
lregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
n	variables
otrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
pregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-2block1/conv2/kernel
 :2block1/conv2/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
t	variables
utrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
vregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
x	variables
ytrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
zregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
|	variables
}trainable_variables
 layer_metrics
‘metrics
’layers
 £layer_regularization_losses
~regularization_losses
€non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
₯layer_metrics
¦metrics
§layers
 ¨layer_regularization_losses
regularization_losses
©non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-2block2/conv1/kernel
 :2block2/conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
ͺlayer_metrics
«metrics
¬layers
 ­layer_regularization_losses
regularization_losses
?non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
―layer_metrics
°metrics
±layers
 ²layer_regularization_losses
regularization_losses
³non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-2block2/conv2/kernel
 :2block2/conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
΄layer_metrics
΅metrics
Άlayers
 ·layer_regularization_losses
regularization_losses
Έnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
6:42block2/convshortcut/kernel
':%2block2/convshortcut/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Ήlayer_metrics
Ίmetrics
»layers
 Όlayer_regularization_losses
regularization_losses
½non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Ύlayer_metrics
Ώmetrics
ΐlayers
 Αlayer_regularization_losses
regularization_losses
Βnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Γlayer_metrics
Δmetrics
Εlayers
 Ζlayer_regularization_losses
 regularization_losses
Ηnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
’	variables
£trainable_variables
Θlayer_metrics
Ιmetrics
Κlayers
 Λlayer_regularization_losses
€regularization_losses
Μnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¦	variables
§trainable_variables
Νlayer_metrics
Ξmetrics
Οlayers
 Πlayer_regularization_losses
¨regularization_losses
Ρnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-2block3/conv1/kernel
 :2block3/conv1/bias
0
ͺ0
«1"
trackable_list_wrapper
0
ͺ0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¬	variables
­trainable_variables
?layer_metrics
Σmetrics
Τlayers
 Υlayer_regularization_losses
?regularization_losses
Φnon_trainable_variables
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
°	variables
±trainable_variables
Χlayer_metrics
Ψmetrics
Ωlayers
 Ϊlayer_regularization_losses
²regularization_losses
Ϋnon_trainable_variables
’__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
/:-2block3/conv2/kernel
 :2block3/conv2/bias
0
΄0
΅1"
trackable_list_wrapper
0
΄0
΅1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ά	variables
·trainable_variables
άlayer_metrics
έmetrics
ήlayers
 ίlayer_regularization_losses
Έregularization_losses
ΰnon_trainable_variables
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
6:42block3/convshortcut/kernel
':%2block3/convshortcut/bias
0
Ί0
»1"
trackable_list_wrapper
0
Ί0
»1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ό	variables
½trainable_variables
αlayer_metrics
βmetrics
γlayers
 δlayer_regularization_losses
Ύregularization_losses
εnon_trainable_variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ΐ	variables
Αtrainable_variables
ζlayer_metrics
ηmetrics
θlayers
 ιlayer_regularization_losses
Βregularization_losses
κnon_trainable_variables
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Δ	variables
Εtrainable_variables
λlayer_metrics
μmetrics
νlayers
 ξlayer_regularization_losses
Ζregularization_losses
οnon_trainable_variables
ͺ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Θ	variables
Ιtrainable_variables
πlayer_metrics
ρmetrics
ςlayers
 σlayer_regularization_losses
Κregularization_losses
τnon_trainable_variables
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Μ	variables
Νtrainable_variables
υlayer_metrics
φmetrics
χlayers
 ψlayer_regularization_losses
Ξregularization_losses
ωnon_trainable_variables
?__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
_generic_user_object
/:-2block4/conv1/kernel
 :2block4/conv1/bias
0
Π0
Ρ1"
trackable_list_wrapper
0
Π0
Ρ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?	variables
Σtrainable_variables
ϊlayer_metrics
ϋmetrics
όlayers
 ύlayer_regularization_losses
Τregularization_losses
ώnon_trainable_variables
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Φ	variables
Χtrainable_variables
?layer_metrics
metrics
layers
 layer_regularization_losses
Ψregularization_losses
non_trainable_variables
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
/:-2block4/conv2/kernel
 :2block4/conv2/bias
0
Ϊ0
Ϋ1"
trackable_list_wrapper
0
Ϊ0
Ϋ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ά	variables
έtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
ήregularization_losses
non_trainable_variables
΄__call__
+΅&call_and_return_all_conditional_losses
'΅"call_and_return_conditional_losses"
_generic_user_object
6:42block4/convshortcut/kernel
':%2block4/convshortcut/bias
0
ΰ0
α1"
trackable_list_wrapper
0
ΰ0
α1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
β	variables
γtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
δregularization_losses
non_trainable_variables
Ά__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ζ	variables
ηtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
θregularization_losses
non_trainable_variables
Έ__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
κ	variables
λtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
μregularization_losses
non_trainable_variables
Ί__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ξ	variables
οtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
πregularization_losses
non_trainable_variables
Ό__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ς	variables
σtrainable_variables
layer_metrics
metrics
layers
  layer_regularization_losses
τregularization_losses
‘non_trainable_variables
Ύ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses"
_generic_user_object
/:-2block5/conv1/kernel
 :2block5/conv1/bias
0
φ0
χ1"
trackable_list_wrapper
0
φ0
χ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ψ	variables
ωtrainable_variables
’layer_metrics
£metrics
€layers
 ₯layer_regularization_losses
ϊregularization_losses
¦non_trainable_variables
ΐ__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ό	variables
ύtrainable_variables
§layer_metrics
¨metrics
©layers
 ͺlayer_regularization_losses
ώregularization_losses
«non_trainable_variables
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
/:-2block5/conv2/kernel
 :2block5/conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
¬layer_metrics
­metrics
?layers
 ―layer_regularization_losses
regularization_losses
°non_trainable_variables
Δ__call__
+Ε&call_and_return_all_conditional_losses
'Ε"call_and_return_conditional_losses"
_generic_user_object
6:42block5/convshortcut/kernel
':%2block5/convshortcut/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
±layer_metrics
²metrics
³layers
 ΄layer_regularization_losses
regularization_losses
΅non_trainable_variables
Ζ__call__
+Η&call_and_return_all_conditional_losses
'Η"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Άlayer_metrics
·metrics
Έlayers
 Ήlayer_regularization_losses
regularization_losses
Ίnon_trainable_variables
Θ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
»layer_metrics
Όmetrics
½layers
 Ύlayer_regularization_losses
regularization_losses
Ώnon_trainable_variables
Κ__call__
+Λ&call_and_return_all_conditional_losses
'Λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
ΐlayer_metrics
Αmetrics
Βlayers
 Γlayer_regularization_losses
regularization_losses
Δnon_trainable_variables
Μ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Εlayer_metrics
Ζmetrics
Ηlayers
 Θlayer_regularization_losses
regularization_losses
Ιnon_trainable_variables
Ξ__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
/:-2block6/conv1/kernel
 :2block6/conv1/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
trainable_variables
Κlayer_metrics
Λmetrics
Μlayers
 Νlayer_regularization_losses
 regularization_losses
Ξnon_trainable_variables
Π__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
’	variables
£trainable_variables
Οlayer_metrics
Πmetrics
Ρlayers
 ?layer_regularization_losses
€regularization_losses
Σnon_trainable_variables
?__call__
+Σ&call_and_return_all_conditional_losses
'Σ"call_and_return_conditional_losses"
_generic_user_object
/:-2block6/conv2/kernel
 :2block6/conv2/bias
0
¦0
§1"
trackable_list_wrapper
0
¦0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¨	variables
©trainable_variables
Τlayer_metrics
Υmetrics
Φlayers
 Χlayer_regularization_losses
ͺregularization_losses
Ψnon_trainable_variables
Τ__call__
+Υ&call_and_return_all_conditional_losses
'Υ"call_and_return_conditional_losses"
_generic_user_object
6:42block6/convshortcut/kernel
':%2block6/convshortcut/bias
0
¬0
­1"
trackable_list_wrapper
0
¬0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?	variables
―trainable_variables
Ωlayer_metrics
Ϊmetrics
Ϋlayers
 άlayer_regularization_losses
°regularization_losses
έnon_trainable_variables
Φ__call__
+Χ&call_and_return_all_conditional_losses
'Χ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
²	variables
³trainable_variables
ήlayer_metrics
ίmetrics
ΰlayers
 αlayer_regularization_losses
΄regularization_losses
βnon_trainable_variables
Ψ__call__
+Ω&call_and_return_all_conditional_losses
'Ω"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ά	variables
·trainable_variables
γlayer_metrics
δmetrics
εlayers
 ζlayer_regularization_losses
Έregularization_losses
ηnon_trainable_variables
Ϊ__call__
+Ϋ&call_and_return_all_conditional_losses
'Ϋ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ί	variables
»trainable_variables
θlayer_metrics
ιmetrics
κlayers
 λlayer_regularization_losses
Όregularization_losses
μnon_trainable_variables
ά__call__
+έ&call_and_return_all_conditional_losses
'έ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ύ	variables
Ώtrainable_variables
νlayer_metrics
ξmetrics
οlayers
 πlayer_regularization_losses
ΐregularization_losses
ρnon_trainable_variables
ή__call__
+ί&call_and_return_all_conditional_losses
'ί"call_and_return_conditional_losses"
_generic_user_object
:
2
fc1/kernel
:2fc1/bias
0
Β0
Γ1"
trackable_list_wrapper
0
Β0
Γ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Δ	variables
Εtrainable_variables
ςlayer_metrics
σmetrics
τlayers
 υlayer_regularization_losses
Ζregularization_losses
φnon_trainable_variables
ΰ__call__
+α&call_and_return_all_conditional_losses
'α"call_and_return_conditional_losses"
_generic_user_object
:
2
fc2/kernel
:2fc2/bias
0
Θ0
Ι1"
trackable_list_wrapper
0
Θ0
Ι1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Κ	variables
Λtrainable_variables
χlayer_metrics
ψmetrics
ωlayers
 ϊlayer_regularization_losses
Μregularization_losses
ϋnon_trainable_variables
β__call__
+γ&call_and_return_all_conditional_losses
'γ"call_and_return_conditional_losses"
_generic_user_object
:
2
fc3/kernel
:2fc3/bias
0
Ξ0
Ο1"
trackable_list_wrapper
0
Ξ0
Ο1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Π	variables
Ρtrainable_variables
όlayer_metrics
ύmetrics
ώlayers
 ?layer_regularization_losses
?regularization_losses
non_trainable_variables
δ__call__
+ε&call_and_return_all_conditional_losses
'ε"call_and_return_conditional_losses"
_generic_user_object
 :	2fc_out/kernel
:2fc_out/bias
0
Τ0
Υ1"
trackable_list_wrapper
0
Τ0
Υ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Φ	variables
Χtrainable_variables
layer_metrics
metrics
layers
 layer_regularization_losses
Ψregularization_losses
non_trainable_variables
ζ__call__
+η&call_and_return_all_conditional_losses
'η"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
ξ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ψ

total

count

_fn_kwargs
	variables
	keras_api"¬
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
-:+H2Adam/conv01/kernel/m
:2Adam/conv01/bias/m
4:22Adam/block0/conv1/kernel/m
%:#2Adam/block0/conv1/bias/m
4:22Adam/block0/conv2/kernel/m
%:#2Adam/block0/conv2/bias/m
4:22Adam/block1/conv1/kernel/m
%:#2Adam/block1/conv1/bias/m
4:22Adam/block1/conv2/kernel/m
%:#2Adam/block1/conv2/bias/m
4:22Adam/block2/conv1/kernel/m
%:#2Adam/block2/conv1/bias/m
4:22Adam/block2/conv2/kernel/m
%:#2Adam/block2/conv2/bias/m
;:92!Adam/block2/convshortcut/kernel/m
,:*2Adam/block2/convshortcut/bias/m
4:22Adam/block3/conv1/kernel/m
%:#2Adam/block3/conv1/bias/m
4:22Adam/block3/conv2/kernel/m
%:#2Adam/block3/conv2/bias/m
;:92!Adam/block3/convshortcut/kernel/m
,:*2Adam/block3/convshortcut/bias/m
4:22Adam/block4/conv1/kernel/m
%:#2Adam/block4/conv1/bias/m
4:22Adam/block4/conv2/kernel/m
%:#2Adam/block4/conv2/bias/m
;:92!Adam/block4/convshortcut/kernel/m
,:*2Adam/block4/convshortcut/bias/m
4:22Adam/block5/conv1/kernel/m
%:#2Adam/block5/conv1/bias/m
4:22Adam/block5/conv2/kernel/m
%:#2Adam/block5/conv2/bias/m
;:92!Adam/block5/convshortcut/kernel/m
,:*2Adam/block5/convshortcut/bias/m
4:22Adam/block6/conv1/kernel/m
%:#2Adam/block6/conv1/bias/m
4:22Adam/block6/conv2/kernel/m
%:#2Adam/block6/conv2/bias/m
;:92!Adam/block6/convshortcut/kernel/m
,:*2Adam/block6/convshortcut/bias/m
#:!
2Adam/fc1/kernel/m
:2Adam/fc1/bias/m
#:!
2Adam/fc2/kernel/m
:2Adam/fc2/bias/m
#:!
2Adam/fc3/kernel/m
:2Adam/fc3/bias/m
%:#	2Adam/fc_out/kernel/m
:2Adam/fc_out/bias/m
-:+H2Adam/conv01/kernel/v
:2Adam/conv01/bias/v
4:22Adam/block0/conv1/kernel/v
%:#2Adam/block0/conv1/bias/v
4:22Adam/block0/conv2/kernel/v
%:#2Adam/block0/conv2/bias/v
4:22Adam/block1/conv1/kernel/v
%:#2Adam/block1/conv1/bias/v
4:22Adam/block1/conv2/kernel/v
%:#2Adam/block1/conv2/bias/v
4:22Adam/block2/conv1/kernel/v
%:#2Adam/block2/conv1/bias/v
4:22Adam/block2/conv2/kernel/v
%:#2Adam/block2/conv2/bias/v
;:92!Adam/block2/convshortcut/kernel/v
,:*2Adam/block2/convshortcut/bias/v
4:22Adam/block3/conv1/kernel/v
%:#2Adam/block3/conv1/bias/v
4:22Adam/block3/conv2/kernel/v
%:#2Adam/block3/conv2/bias/v
;:92!Adam/block3/convshortcut/kernel/v
,:*2Adam/block3/convshortcut/bias/v
4:22Adam/block4/conv1/kernel/v
%:#2Adam/block4/conv1/bias/v
4:22Adam/block4/conv2/kernel/v
%:#2Adam/block4/conv2/bias/v
;:92!Adam/block4/convshortcut/kernel/v
,:*2Adam/block4/convshortcut/bias/v
4:22Adam/block5/conv1/kernel/v
%:#2Adam/block5/conv1/bias/v
4:22Adam/block5/conv2/kernel/v
%:#2Adam/block5/conv2/bias/v
;:92!Adam/block5/convshortcut/kernel/v
,:*2Adam/block5/convshortcut/bias/v
4:22Adam/block6/conv1/kernel/v
%:#2Adam/block6/conv1/bias/v
4:22Adam/block6/conv2/kernel/v
%:#2Adam/block6/conv2/bias/v
;:92!Adam/block6/convshortcut/kernel/v
,:*2Adam/block6/convshortcut/bias/v
#:!
2Adam/fc1/kernel/v
:2Adam/fc1/bias/v
#:!
2Adam/fc2/kernel/v
:2Adam/fc2/bias/v
#:!
2Adam/fc3/kernel/v
:2Adam/fc3/bias/v
%:#	2Adam/fc_out/kernel/v
:2Adam/fc_out/bias/v
ξ2λ
(__inference_MTFNet_layer_call_fn_3747298
(__inference_MTFNet_layer_call_fn_3746554
(__inference_MTFNet_layer_call_fn_3747399
(__inference_MTFNet_layer_call_fn_3746295ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ϊ2Χ
C__inference_MTFNet_layer_call_and_return_conditional_losses_3747197
C__inference_MTFNet_layer_call_and_return_conditional_losses_3745877
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746931
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746035ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
θ2ε
"__inference__wrapped_model_3744708Ύ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *.’+
)&
input_1?????????  H
?2Ο
(__inference_conv01_layer_call_fn_3747418’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_conv01_layer_call_and_return_conditional_losses_3747409’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Φ2Σ
,__inference_activation_layer_call_fn_3747428’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ρ2ξ
G__inference_activation_layer_call_and_return_conditional_losses_3747423’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block0/conv1_layer_call_fn_3747447’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block0/conv1_layer_call_and_return_conditional_losses_3747438’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_1_layer_call_fn_3747457’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_1_layer_call_and_return_conditional_losses_3747452’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block0/conv2_layer_call_fn_3747476’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block0/conv2_layer_call_and_return_conditional_losses_3747467’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_2_layer_call_fn_3747486’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_2_layer_call_and_return_conditional_losses_3747481’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_add_layer_call_fn_3747498’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_add_layer_call_and_return_conditional_losses_3747492’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_3_layer_call_fn_3747508’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_3_layer_call_and_return_conditional_losses_3747503’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block1/conv1_layer_call_fn_3747527’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block1/conv1_layer_call_and_return_conditional_losses_3747518’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_4_layer_call_fn_3747537’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_4_layer_call_and_return_conditional_losses_3747532’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block1/conv2_layer_call_fn_3747556’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block1/conv2_layer_call_and_return_conditional_losses_3747547’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_5_layer_call_fn_3747566’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_5_layer_call_and_return_conditional_losses_3747561’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_1_layer_call_fn_3747578’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_1_layer_call_and_return_conditional_losses_3747572’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_6_layer_call_fn_3747588’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_6_layer_call_and_return_conditional_losses_3747583’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block2/conv1_layer_call_fn_3747607’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block2/conv1_layer_call_and_return_conditional_losses_3747598’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_7_layer_call_fn_3747617’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_7_layer_call_and_return_conditional_losses_3747612’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block2/conv2_layer_call_fn_3747636’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block2/conv2_layer_call_and_return_conditional_losses_3747627’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2ά
5__inference_block2/convshortcut_layer_call_fn_3747655’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊ2χ
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_3747646’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_8_layer_call_fn_3747665’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_8_layer_call_and_return_conditional_losses_3747660’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_activation_9_layer_call_fn_3747675’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_activation_9_layer_call_and_return_conditional_losses_3747670’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_2_layer_call_fn_3747687’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_2_layer_call_and_return_conditional_losses_3747681’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_10_layer_call_fn_3747697’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_10_layer_call_and_return_conditional_losses_3747692’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block3/conv1_layer_call_fn_3747716’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block3/conv1_layer_call_and_return_conditional_losses_3747707’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_11_layer_call_fn_3747726’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_11_layer_call_and_return_conditional_losses_3747721’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block3/conv2_layer_call_fn_3747745’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block3/conv2_layer_call_and_return_conditional_losses_3747736’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2ά
5__inference_block3/convshortcut_layer_call_fn_3747764’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊ2χ
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_3747755’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_12_layer_call_fn_3747774’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_12_layer_call_and_return_conditional_losses_3747769’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_13_layer_call_fn_3747784’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_13_layer_call_and_return_conditional_losses_3747779’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_3_layer_call_fn_3747796’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_3_layer_call_and_return_conditional_losses_3747790’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_14_layer_call_fn_3747806’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_14_layer_call_and_return_conditional_losses_3747801’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block4/conv1_layer_call_fn_3747825’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block4/conv1_layer_call_and_return_conditional_losses_3747816’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_15_layer_call_fn_3747835’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_15_layer_call_and_return_conditional_losses_3747830’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block4/conv2_layer_call_fn_3747854’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block4/conv2_layer_call_and_return_conditional_losses_3747845’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2ά
5__inference_block4/convshortcut_layer_call_fn_3747873’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊ2χ
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_3747864’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_16_layer_call_fn_3747883’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_16_layer_call_and_return_conditional_losses_3747878’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_17_layer_call_fn_3747893’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_17_layer_call_and_return_conditional_losses_3747888’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_4_layer_call_fn_3747905’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_4_layer_call_and_return_conditional_losses_3747899’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_18_layer_call_fn_3747915’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_18_layer_call_and_return_conditional_losses_3747910’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block5/conv1_layer_call_fn_3747934’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block5/conv1_layer_call_and_return_conditional_losses_3747925’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_19_layer_call_fn_3747944’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_19_layer_call_and_return_conditional_losses_3747939’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block5/conv2_layer_call_fn_3747963’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block5/conv2_layer_call_and_return_conditional_losses_3747954’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2ά
5__inference_block5/convshortcut_layer_call_fn_3747982’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊ2χ
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_3747973’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_20_layer_call_fn_3747992’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_20_layer_call_and_return_conditional_losses_3747987’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_21_layer_call_fn_3748002’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_21_layer_call_and_return_conditional_losses_3747997’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_5_layer_call_fn_3748014’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_5_layer_call_and_return_conditional_losses_3748008’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_22_layer_call_fn_3748024’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_22_layer_call_and_return_conditional_losses_3748019’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block6/conv1_layer_call_fn_3748043’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block6/conv1_layer_call_and_return_conditional_losses_3748034’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_23_layer_call_fn_3748053’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_23_layer_call_and_return_conditional_losses_3748048’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ψ2Υ
.__inference_block6/conv2_layer_call_fn_3748072’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_block6/conv2_layer_call_and_return_conditional_losses_3748063’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2ά
5__inference_block6/convshortcut_layer_call_fn_3748091’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊ2χ
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_3748082’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_24_layer_call_fn_3748101’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_24_layer_call_and_return_conditional_losses_3748096’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_25_layer_call_fn_3748111’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_25_layer_call_and_return_conditional_losses_3748106’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_add_6_layer_call_fn_3748123’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_add_6_layer_call_and_return_conditional_losses_3748117’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ω2Φ
/__inference_activation_26_layer_call_fn_3748133’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ2ρ
J__inference_activation_26_layer_call_and_return_conditional_losses_3748128’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_fc1_layer_call_fn_3748173’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_fc1_layer_call_and_return_conditional_losses_3748164’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_fc2_layer_call_fn_3748213’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_fc2_layer_call_and_return_conditional_losses_3748204’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ο2Μ
%__inference_fc3_layer_call_fn_3748253’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_fc3_layer_call_and_return_conditional_losses_3748244’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2Ο
(__inference_fc_out_layer_call_fn_3748293’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_fc_out_layer_call_and_return_conditional_losses_3748284’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
4B2
%__inference_signature_wrapper_3746665input_1
C__inference_MTFNet_layer_call_and_return_conditional_losses_3745877ΙVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ@’=
6’3
)&
input_1?????????  H
p

 
ͺ "-’*
# 
0?????????
 
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746035ΙVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ@’=
6’3
)&
input_1?????????  H
p 

 
ͺ "-’*
# 
0?????????
 
C__inference_MTFNet_layer_call_and_return_conditional_losses_3746931ΘVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ?’<
5’2
(%
inputs?????????  H
p

 
ͺ "-’*
# 
0?????????
 
C__inference_MTFNet_layer_call_and_return_conditional_losses_3747197ΘVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ?’<
5’2
(%
inputs?????????  H
p 

 
ͺ "-’*
# 
0?????????
 ι
(__inference_MTFNet_layer_call_fn_3746295ΌVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ@’=
6’3
)&
input_1?????????  H
p

 
ͺ " ?????????ι
(__inference_MTFNet_layer_call_fn_3746554ΌVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ@’=
6’3
)&
input_1?????????  H
p 

 
ͺ " ?????????θ
(__inference_MTFNet_layer_call_fn_3747298»VBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ?’<
5’2
(%
inputs?????????  H
p

 
ͺ " ?????????θ
(__inference_MTFNet_layer_call_fn_3747399»VBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ?’<
5’2
(%
inputs?????????  H
p 

 
ͺ " ?????????ς
"__inference__wrapped_model_3744708ΛVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥ8’5
.’+
)&
input_1?????????  H
ͺ "7ͺ4
2
fc_out(%
fc_out?????????Έ
J__inference_activation_10_layer_call_and_return_conditional_losses_3747692j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_10_layer_call_fn_3747697]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_11_layer_call_and_return_conditional_losses_3747721j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_11_layer_call_fn_3747726]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_12_layer_call_and_return_conditional_losses_3747769j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_12_layer_call_fn_3747774]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_13_layer_call_and_return_conditional_losses_3747779j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_13_layer_call_fn_3747784]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_14_layer_call_and_return_conditional_losses_3747801j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_14_layer_call_fn_3747806]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_15_layer_call_and_return_conditional_losses_3747830j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_15_layer_call_fn_3747835]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_16_layer_call_and_return_conditional_losses_3747878j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_16_layer_call_fn_3747883]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_17_layer_call_and_return_conditional_losses_3747888j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_17_layer_call_fn_3747893]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_18_layer_call_and_return_conditional_losses_3747910j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_18_layer_call_fn_3747915]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_19_layer_call_and_return_conditional_losses_3747939j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_19_layer_call_fn_3747944]8’5
.’+
)&
inputs?????????
ͺ "!?????????·
I__inference_activation_1_layer_call_and_return_conditional_losses_3747452j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_1_layer_call_fn_3747457]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  Έ
J__inference_activation_20_layer_call_and_return_conditional_losses_3747987j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_20_layer_call_fn_3747992]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_21_layer_call_and_return_conditional_losses_3747997j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_21_layer_call_fn_3748002]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_22_layer_call_and_return_conditional_losses_3748019j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_22_layer_call_fn_3748024]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_23_layer_call_and_return_conditional_losses_3748048j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_23_layer_call_fn_3748053]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_24_layer_call_and_return_conditional_losses_3748096j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_24_layer_call_fn_3748101]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_25_layer_call_and_return_conditional_losses_3748106j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_25_layer_call_fn_3748111]8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
J__inference_activation_26_layer_call_and_return_conditional_losses_3748128j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
/__inference_activation_26_layer_call_fn_3748133]8’5
.’+
)&
inputs?????????
ͺ "!?????????·
I__inference_activation_2_layer_call_and_return_conditional_losses_3747481j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_2_layer_call_fn_3747486]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ·
I__inference_activation_3_layer_call_and_return_conditional_losses_3747503j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_3_layer_call_fn_3747508]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ·
I__inference_activation_4_layer_call_and_return_conditional_losses_3747532j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_4_layer_call_fn_3747537]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ·
I__inference_activation_5_layer_call_and_return_conditional_losses_3747561j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_5_layer_call_fn_3747566]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ·
I__inference_activation_6_layer_call_and_return_conditional_losses_3747583j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_activation_6_layer_call_fn_3747588]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ·
I__inference_activation_7_layer_call_and_return_conditional_losses_3747612j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_activation_7_layer_call_fn_3747617]8’5
.’+
)&
inputs?????????
ͺ "!?????????·
I__inference_activation_8_layer_call_and_return_conditional_losses_3747660j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_activation_8_layer_call_fn_3747665]8’5
.’+
)&
inputs?????????
ͺ "!?????????·
I__inference_activation_9_layer_call_and_return_conditional_losses_3747670j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_activation_9_layer_call_fn_3747675]8’5
.’+
)&
inputs?????????
ͺ "!?????????΅
G__inference_activation_layer_call_and_return_conditional_losses_3747423j8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
,__inference_activation_layer_call_fn_3747428]8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ε
B__inference_add_1_layer_call_and_return_conditional_losses_3747572l’i
b’_
]Z
+(
inputs/0?????????  
+(
inputs/1?????????  
ͺ ".’+
$!
0?????????  
 ½
'__inference_add_1_layer_call_fn_3747578l’i
b’_
]Z
+(
inputs/0?????????  
+(
inputs/1?????????  
ͺ "!?????????  ε
B__inference_add_2_layer_call_and_return_conditional_losses_3747681l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ½
'__inference_add_2_layer_call_fn_3747687l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ "!?????????ε
B__inference_add_3_layer_call_and_return_conditional_losses_3747790l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ½
'__inference_add_3_layer_call_fn_3747796l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ "!?????????ε
B__inference_add_4_layer_call_and_return_conditional_losses_3747899l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ½
'__inference_add_4_layer_call_fn_3747905l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ "!?????????ε
B__inference_add_5_layer_call_and_return_conditional_losses_3748008l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ½
'__inference_add_5_layer_call_fn_3748014l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ "!?????????ε
B__inference_add_6_layer_call_and_return_conditional_losses_3748117l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ½
'__inference_add_6_layer_call_fn_3748123l’i
b’_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ͺ "!?????????γ
@__inference_add_layer_call_and_return_conditional_losses_3747492l’i
b’_
]Z
+(
inputs/0?????????  
+(
inputs/1?????????  
ͺ ".’+
$!
0?????????  
 »
%__inference_add_layer_call_fn_3747498l’i
b’_
]Z
+(
inputs/0?????????  
+(
inputs/1?????????  
ͺ "!?????????  »
I__inference_block0/conv1_layer_call_and_return_conditional_losses_3747438nLM8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_block0/conv1_layer_call_fn_3747447aLM8’5
.’+
)&
inputs?????????  
ͺ "!?????????  »
I__inference_block0/conv2_layer_call_and_return_conditional_losses_3747467nVW8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_block0/conv2_layer_call_fn_3747476aVW8’5
.’+
)&
inputs?????????  
ͺ "!?????????  »
I__inference_block1/conv1_layer_call_and_return_conditional_losses_3747518nhi8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_block1/conv1_layer_call_fn_3747527ahi8’5
.’+
)&
inputs?????????  
ͺ "!?????????  »
I__inference_block1/conv2_layer_call_and_return_conditional_losses_3747547nrs8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????  
 
.__inference_block1/conv2_layer_call_fn_3747556ars8’5
.’+
)&
inputs?????????  
ͺ "!?????????  ½
I__inference_block2/conv1_layer_call_and_return_conditional_losses_3747598p8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????
 
.__inference_block2/conv1_layer_call_fn_3747607c8’5
.’+
)&
inputs?????????  
ͺ "!?????????½
I__inference_block2/conv2_layer_call_and_return_conditional_losses_3747627p8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block2/conv2_layer_call_fn_3747636c8’5
.’+
)&
inputs?????????
ͺ "!?????????Δ
P__inference_block2/convshortcut_layer_call_and_return_conditional_losses_3747646p8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????
 
5__inference_block2/convshortcut_layer_call_fn_3747655c8’5
.’+
)&
inputs?????????  
ͺ "!?????????½
I__inference_block3/conv1_layer_call_and_return_conditional_losses_3747707pͺ«8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block3/conv1_layer_call_fn_3747716cͺ«8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block3/conv2_layer_call_and_return_conditional_losses_3747736p΄΅8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block3/conv2_layer_call_fn_3747745c΄΅8’5
.’+
)&
inputs?????????
ͺ "!?????????Δ
P__inference_block3/convshortcut_layer_call_and_return_conditional_losses_3747755pΊ»8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
5__inference_block3/convshortcut_layer_call_fn_3747764cΊ»8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block4/conv1_layer_call_and_return_conditional_losses_3747816pΠΡ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block4/conv1_layer_call_fn_3747825cΠΡ8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block4/conv2_layer_call_and_return_conditional_losses_3747845pΪΫ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block4/conv2_layer_call_fn_3747854cΪΫ8’5
.’+
)&
inputs?????????
ͺ "!?????????Δ
P__inference_block4/convshortcut_layer_call_and_return_conditional_losses_3747864pΰα8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
5__inference_block4/convshortcut_layer_call_fn_3747873cΰα8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block5/conv1_layer_call_and_return_conditional_losses_3747925pφχ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block5/conv1_layer_call_fn_3747934cφχ8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block5/conv2_layer_call_and_return_conditional_losses_3747954p8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block5/conv2_layer_call_fn_3747963c8’5
.’+
)&
inputs?????????
ͺ "!?????????Δ
P__inference_block5/convshortcut_layer_call_and_return_conditional_losses_3747973p8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
5__inference_block5/convshortcut_layer_call_fn_3747982c8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block6/conv1_layer_call_and_return_conditional_losses_3748034p8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block6/conv1_layer_call_fn_3748043c8’5
.’+
)&
inputs?????????
ͺ "!?????????½
I__inference_block6/conv2_layer_call_and_return_conditional_losses_3748063p¦§8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
.__inference_block6/conv2_layer_call_fn_3748072c¦§8’5
.’+
)&
inputs?????????
ͺ "!?????????Δ
P__inference_block6/convshortcut_layer_call_and_return_conditional_losses_3748082p¬­8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
5__inference_block6/convshortcut_layer_call_fn_3748091c¬­8’5
.’+
)&
inputs?????????
ͺ "!?????????΄
C__inference_conv01_layer_call_and_return_conditional_losses_3747409mBC7’4
-’*
(%
inputs?????????  H
ͺ ".’+
$!
0?????????  
 
(__inference_conv01_layer_call_fn_3747418`BC7’4
-’*
(%
inputs?????????  H
ͺ "!?????????  ΄
@__inference_fc1_layer_call_and_return_conditional_losses_3748164pΒΓ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
%__inference_fc1_layer_call_fn_3748173cΒΓ8’5
.’+
)&
inputs?????????
ͺ "!?????????΄
@__inference_fc2_layer_call_and_return_conditional_losses_3748204pΘΙ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
%__inference_fc2_layer_call_fn_3748213cΘΙ8’5
.’+
)&
inputs?????????
ͺ "!?????????΄
@__inference_fc3_layer_call_and_return_conditional_losses_3748244pΞΟ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
%__inference_fc3_layer_call_fn_3748253cΞΟ8’5
.’+
)&
inputs?????????
ͺ "!?????????Ά
C__inference_fc_out_layer_call_and_return_conditional_losses_3748284oΤΥ8’5
.’+
)&
inputs?????????
ͺ "-’*
# 
0?????????
 
(__inference_fc_out_layer_call_fn_3748293bΤΥ8’5
.’+
)&
inputs?????????
ͺ " ?????????
%__inference_signature_wrapper_3746665ΦVBCLMVWhirsͺ«Ί»΄΅ΠΡΰαΪΫφχ¬­¦§ΒΓΘΙΞΟΤΥC’@
’ 
9ͺ6
4
input_1)&
input_1?????????  H"7ͺ4
2
fc_out(%
fc_out?????????