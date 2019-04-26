class Bias{
  int layer;
  float bias=1;
  float weight;
  int x;
  int y;
  int r=30;
  float value;
  
  Bias(int layer_,float weight_){
    this.layer=layer_;
    this.weight=weight_;
    Layer currLayer=l[this.layer];
    this.value=this.bias*this.weight;
    this.x=xo+150*(currLayer.layerId);
    this.y=yo+150*(currLayer.nbNeurons);
  }
  
  void show(){
    //bias
    int cv=(int)map(this.bias,-1,1,0,255);
    int c=color(255-cv,cv,0);
    fill(c);
    ellipse(this.x,this.y,this.r,this.r);
    //bias value
    fill(255);
    textSize(ts);
    text(this.bias,this.x-30,this.y-30);
  }
}
