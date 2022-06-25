import logo from './logo.svg';
import './App.css';
import ImageUploader from './components/imageUploader';
import { useEffect, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import {CircularProgress, Button, Stack, Input, IconButton } from '@mui/material';
import * as React from 'react';



import Resizer from 'react-image-file-resizer';
import { flexbox } from '@mui/system';

function App() {

  const classes =  [
    "Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vechiles over 3.5 metric tons","Right-of-way at the next intersection","Priority road","Yield","Stop","No vechiles","Vechiles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vechiles over 3.5 metric tons"
  ]; 
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(null);
  const [processedImage, setProcessedImage] = useState({
    src: '',
    err: ''
  })
  const [prediction, setPrediction] = useState({
    loading: false,
    result: ''
  });
  
  const imageRef = useRef();
  const visibleImageRef = useRef();

  const loadModel = async () => {

    setLoading(true);
    const model = await tf.loadLayersModel('http://localhost/cnn_model/model.json');
    setModel(model);
    console.log(model);
    console.log("Model loaded");
    setLoading(false);

  }

  const imageStyle = {
   display: 'none'
  }
  const buttonStyle = {
    
  }
  const AppStyle = {
    display: 'grid',
    justifyItems: 'center',
    alignItems: 'center'
  }
 
  useEffect(() => {
    loadModel();
  }, [])

  const UploadButton = () => {
    return (
      <div>
      <label htmlFor="contained-button-file">
        <Input accept="image/*" id="contained-button-file" multiple type="file"  onChange={async (e) => {
          await resizeFile(e.target.files[0]); 
          setImage(e.target.files[0]);
        }} ></Input>
        
      </label>
      </div>
     
    )
  }

  const resizeFile = (file) => new Promise(resolve => {
    
    Resizer.imageFileResizer(file, 32, 32, 'JPEG', 100, 0,
    uri => {
      resolve(uri);
      setProcessedImage(uri); 
    },
    'base64'
    );

});
  const classifyImage = async () => {
    let tensor = tf.browser.fromPixels(imageRef.current).resizeBilinear([32, 32]).toFloat().expandDims().mean(-1).expandDims(3)
   console.log(tensor);
    
    //const expandedDims = tf.expandDims(tensor); //expand to 4 dimensions (batch, height, width, channels)
    //const reshapeTensor = tensor.reshape([1, 32, 32, 3]); //reshape to (batch, height, width, channels)
    //console.log(reshapeTensor);
    setPrediction({...prediction, loading: true}); 
    let resp = await model.predict(tensor).data();
    console.log(resp);
    if(resp.length > 0){
      let top5 = Array.from(resp).map((item, index) => {
        return {
          prob: item,
          classIndex: index,
          className: classes[index]
        }
      }).sort(function (a, b){
        return b.prob - a.prob;
      }); 
      console.log(top5);
      setPrediction({
        loading: false,
        result: top5[0].className
      })
    }

  }

  return (
    <div className="App" style={AppStyle}>
      <div>
        {loading === true ? (<div style={{textAlign: 'center', display: 'flex', justifyItems: 'center', flexDirection: 'column'}}>
          <h1>Ładowanie modelu...</h1>
          <CircularProgress/>
          </div>) : (
          <div>
          <h1>Upload an image</h1>
          <div style={buttonStyle}>
            <UploadButton/>
          </div>

       
            {image &&
              (
                <div>

                  <div>
                    
                    <img src={processedImage} alt="" style={imageStyle}
                    ref={imageRef} />
                    

                  </div>
                  <div className="fullscale-image">
                    
                      <img src={URL.createObjectURL(image)} alt="" ref={visibleImageRef}/>
                    
                  </div>
                  <div style={{textAlign: 'center', marginTop: '40px'}}>
                    <Button variant="contained" onClick={classifyImage}>Classify</Button>
                    {prediction.loading ? (<CircularProgress/>) : (<h1>{prediction.result}</h1>)}
                    
                  </div>
                </div>

              )

            }
            
       

          </div>
        )
        }
      </div>
    </div>




  );
}

export default App;
