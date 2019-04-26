class Neuron{
  int id;
  int x;
  int y;
  int r=30;  //radius
  int ts=15; //text size
  int layer;
  float value;
  float[] weights;
  float bias;
  
  Neuron(int layer_,int id_){
    this.id=id_;
    this.layer=layer_;
    this.x=xo+150*layer;
    this.y=yo+150*id;
    
    if (this.layer==0){
      this.value=inputs[this.id];
    } else {
      feedForward();
    }
  }
  
  void feedForward(){
    if (this.layer==0){
      //feed inputs
      this.value=inputs[this.id];
    } else {
      //calc next outputs per layer
      Layer prevLayer=l[this.layer-1];
      float[] inputs=new float[prevLayer.nbNeurons];
      float[] weights=new float[prevLayer.nbNeurons];
      for (int i=0;i<prevLayer.nbNeurons;i++){
        inputs[i]=prevLayer.n[i].value;
        weights[i]=Weights[prevLayer.layerId][i][this.id];
      }
      this.weights=weights;
      this.bias=prevLayer.b.value;
      this.value=net(inputs,weights,bias);
      this.value=out(this.value);
    }
  }
  
  void show(){
    feedForward();
    //neuron
    int cv=(int) map(this.value,-1,1,0,255);
    int c=color(255-cv,cv,0);
    fill(c);
    ellipse(this.x,this.y,this.r,this.r);
    //neuron value
    fill(255);
    textSize(ts);
    text(this.value,this.x-30,this.y-30);
    //connection weights to the neuron
    if (layer>0){
      Layer prevLayer=l[this.layer-1];
      for (int i=0;i<prevLayer.nbNeurons;i++){
        int cw=(int) map(this.weights[i],-1,1,0,255);
        c=color(255-cw,cw,0); 
        stroke(c);
        line(prevLayer.n[i].x,prevLayer.n[i].y,this.x,this.y);
        line(prevLayer.b.x,prevLayer.b.y,this.x,this.y);
        //weight value
        stroke(255);
        ellipse((prevLayer.n[i].x*(1-this.weights[i])+this.x*(1+this.weights[i]))/2,
                (prevLayer.n[i].y*(1-this.weights[i])+this.y*(1+this.weights[i]))/2,
                5,5);
      }
      ellipse((prevLayer.b.x*(1-prevLayer.b.weight)+this.x*(1+prevLayer.b.weight))/2,
              (prevLayer.b.y*(1-prevLayer.b.weight)+this.y*(1+prevLayer.b.weight))/2,
              5,5);
    }
  }
  
}
