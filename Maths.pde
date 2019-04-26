float net(float[] inputs,float[] weights,float bias){
  float output=0;
  for (int i=0;i<inputs.length;i++)
    output+=inputs[i]*weights[i];
  output+=bias;
  return output;
}

float error(float value,float target){
  return .5*pow((value-target),2);
}

float dError(float value,float target){
  return (value-target);
}

float out(float x){
  return relu(x);
}

float relu(float x){
  if (x<0) x=0;
  return x;
}

float dRelu(float x){
  if (x<0) x=0;
  if (x>=0) x=1;
  return x;
}

float sigmoid(float x){
  float y=1/(1+exp(-x));
  return y;
}

float dSigmoid(float x){
  return x*(1-x);
}
