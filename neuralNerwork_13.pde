int counter=0;
int generation=0;
int xo=50;
int yo=100;
int ts=15;
Layer[] l=new Layer[3];
float[] Inputs={0.05,0.1};
float[][][] Weights={
  {{0.15,0.25},{0.2,0.3}},
  {{0.4,0.5},{0.45,0.55}}
};

float[] targets={0.01,0.99};
float lr=0.5;
float[] inputs={0.05,0.1};
float[] weights_h1={0.15,0.2};
float[] weights_h2={0.25,0.3}; 
float[] weights_o1={0.4,0.45};
float[] weights_o2={0.5,0.55};

void setup(){
  size(800,800);
  //structure neural network
  for (int i=0;i<l.length;i++)
    l[i]=new Layer(i);
  l[0].setNeurons(2);
  l[0].setBias(0.35);
  l[1].setNeurons(2);
  l[1].setBias(0.6);
  l[2].setNeurons(2);  
}

void draw(){
  background(0);
  
  float E_tot=train();
  if (E_tot<0.0001){
    inputs[0]=random(0,1);
    inputs[1]=1-inputs[0];
    targets[0]=inputs[1];
    targets[1]=inputs[0];
    counter=0;
    generation++;
  }
  for (int i=0;i<l.length;i++)
    l[i].show();
}

float train(){
  //forward pass
  counter++;
  float bias_h1=0.35;
  float bias_h2=0.35;
  float bias_o1=0.6;
  float bias_o2=0.6;
  float net_h1=net(inputs,weights_h1,bias_h1);
  float net_h2=net(inputs,weights_h2,bias_h2);
  float out_h1=out(net_h1);
  float out_h2=out(net_h2);
  float[] hidden={out_h1,out_h2};
  float net_o1=net(hidden,weights_o1,bias_o1);
  float net_o2=net(hidden,weights_o2,bias_o2);
  float out_o1=out(net_o1);
  float out_o2=out(net_o2);
  text(out_o1,80,500);
  text(targets[0],150,500);
  text(out_o2,80,550);
  text(targets[1],150,550);
  
  //total error
  //float[] targets={0.01,0.99};
  float E_o1=error(out_o1,targets[0]); 
  float E_o2=error(out_o2,targets[1]); 
  float E_tot=E_o1+E_o2;
  println(E_tot,counter);
  text(generation,20,height-30);
  text(counter,120,height-30);
  text(E_tot*1000,220,height-30);
  fill(0,0,255);
  rect(0,height-20,E_tot*1000*width,height);
  
  //backwards pass
  //output layer
  float dE_tot_dOut_o1=(out_o1-targets[0]);
  float dE_tot_dOut_o2=(out_o2-targets[1]);
  float dOut_o1_dNet_o1=dRelu(out_o1);
  float dOut_o2_dNet_o2=dRelu(out_o2);
  float dNet_o1_dw5=out_h1;
  float dNet_o1_dw6=out_h2;
  float dNet_o1_dbo1=1;
  float dNet_o2_dw7=out_h1;
  float dNet_o2_dw8=out_h2;
  float dNet_o2_dbo2=1;
  float dE_tot_dw5=dE_tot_dOut_o1*dOut_o1_dNet_o1*dNet_o1_dw5;
  float dE_tot_dw6=dE_tot_dOut_o1*dOut_o1_dNet_o1*dNet_o1_dw6;
  float dE_tot_dbo1=dE_tot_dOut_o1*dOut_o1_dNet_o1*dNet_o1_dbo1;
  float dE_tot_dw7=dE_tot_dOut_o2*dOut_o2_dNet_o2*dNet_o2_dw7;
  float dE_tot_dw8=dE_tot_dOut_o2*dOut_o2_dNet_o2*dNet_o2_dw8;
  float dE_tot_dbo2=dE_tot_dOut_o2*dOut_o2_dNet_o2*dNet_o2_dbo2;
  //hidden layer
  float dE_o1_dNet_o1=dE_tot_dOut_o1*dOut_o1_dNet_o1;
  float dE_o2_dNet_o2=dE_tot_dOut_o2*dOut_o2_dNet_o2;
  float dNet_o1_dOut_h1=weights_o1[0];
  float dNet_o2_dOut_h1=weights_o2[0];
  float dNet_o1_dOut_h2=weights_o1[1];
  float dNet_o2_dOut_h2=weights_o2[1];
  float dE_o1_dOut_h1=dE_o1_dNet_o1*dNet_o1_dOut_h1;
  float dE_o2_dOut_h1=dE_o2_dNet_o2*dNet_o2_dOut_h1;
  float dE_tot_dOut_h1=dE_o1_dOut_h1+dE_o2_dOut_h1;
  float dE_o1_dOut_h2=dE_o1_dNet_o1*dNet_o1_dOut_h2;
  float dE_o2_dOut_h2=dE_o2_dNet_o2*dNet_o2_dOut_h2;
  float dE_tot_dOut_h2=dE_o1_dOut_h2+dE_o2_dOut_h2;  
  float dOut_h1_dNet_h1=dRelu(out_h1);
  float dOut_h2_dNet_h2=dRelu(out_h2);
  float dNet_h1_dw1=inputs[0];
  float dNet_h1_dw2=inputs[1];
  float dNet_h1_dbh1=1;
  float dNet_h2_dw3=inputs[0];
  float dNet_h2_dw4=inputs[1];
  float dNet_h2_dbh2=1;
  float dE_tot_dw1=dE_tot_dOut_h1*dOut_h1_dNet_h1*dNet_h1_dw1;
  float dE_tot_dw2=dE_tot_dOut_h1*dOut_h1_dNet_h1*dNet_h1_dw2;
  float dE_tot_dbh1=dE_tot_dOut_h1*dOut_h1_dNet_h1*dNet_h1_dbh1;
  float dE_tot_dw3=dE_tot_dOut_h2*dOut_h2_dNet_h2*dNet_h2_dw3;
  float dE_tot_dw4=dE_tot_dOut_h2*dOut_h2_dNet_h2*dNet_h2_dw4;
  float dE_tot_dbh2=dE_tot_dOut_h2*dOut_h2_dNet_h2*dNet_h2_dbh2;
  
  //update weights
  weights_h1[0]=(weights_h1[0]-lr*dE_tot_dw1);
  weights_h1[1]=(weights_h1[1]-lr*dE_tot_dw2);
  bias_h1=bias_h1-lr*dE_tot_dbh1;
  weights_h2[0]=(weights_h2[0]-lr*dE_tot_dw3);
  weights_h2[1]=(weights_h2[1]-lr*dE_tot_dw4);
  bias_h2=bias_h2-lr*dE_tot_dbh2;
  weights_o1[0]=(weights_o1[0]-lr*dE_tot_dw5);
  weights_o1[1]=(weights_o1[1]-lr*dE_tot_dw6);
  bias_o1=bias_o1-lr*dE_tot_dbo1;
  weights_o2[0]=(weights_o2[0]-lr*dE_tot_dw7);
  weights_o2[1]=(weights_o2[1]-lr*dE_tot_dw8);  
  bias_o2=bias_o2-lr*dE_tot_dbo2;
  print(weights_h1[0],weights_h1[1],weights_h2[0],weights_h2[1]," ");
  println(weights_o1[0],weights_o1[1],weights_o2[0],weights_o2[1]);
  
  Weights[0][0][0]=weights_h1[0];
  Weights[0][0][1]=weights_h1[1];
  l[0].setBias(bias_h1);
  Weights[0][1][0]=weights_h2[0];
  Weights[0][1][1]=weights_h2[1];
  l[0].setBias(bias_h2);
  Weights[1][0][0]=weights_o1[0];
  Weights[1][0][1]=weights_o1[1];
  l[1].setBias(bias_o1);
  Weights[1][1][0]=weights_o2[0];
  Weights[1][1][1]=weights_o2[1];
  l[1].setBias(bias_o2);
  
  return E_tot;
}
