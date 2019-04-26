class Layer{
  int layerId;
  int nbNeurons;
  Neuron[] n;
  Bias b;
  
  Layer(int layerId_){
    this.layerId=layerId_;
  }
  
  void setNeurons(int nbNeurons_){
    this.nbNeurons=nbNeurons_;
    n=new Neuron[this.nbNeurons];
    for (int i=0;i<this.nbNeurons;i++)
      n[i]=new Neuron(this.layerId,i);
  }
  
  void setBias(float weight_){
    b=new Bias(this.layerId,weight_);
  }
  
  void show(){
    for (int i=0;i<this.nbNeurons;i++)
      n[i].show();
    if (this.layerId<2)
      b.show();
  }
}
