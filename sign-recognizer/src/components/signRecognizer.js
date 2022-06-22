import React, {useState, useEffect} from 'react';
import * as tf from '@tensorflow/tfjs'





const SignRecognizer = async () => {
    const [sign, setSign] = useState(null);
    const [loading, setLoading] = useState(false);
    const [model, setModel] = useState(null); 
   
    const loadModel = async () => {
      setLoading(true); 
      try {
        const model = await tf.loadLayersModel('../model/TSC_json_model.json');
        console.log(model);
        setModel(model); 
        setLoading(false); 
      } catch(err){
        console.log(err); 
        setLoading(false); 
      }
      
  }
  
  
    useEffect(()=> {
      loadModel();
      setSign(this.props.sign);
    })

    const recognize = async () => {
        const response = await model.predict(sign);
        console.log(response); 
        return response; 

    }
    if(loading){
      return (
        <div>
          <h1>Loading...</h1>
        </div>
      )
    }


  return (
    <div>
        
    </div>
  )
}

export default SignRecognizer