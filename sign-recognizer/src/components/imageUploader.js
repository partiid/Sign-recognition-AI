import React, {useState, useEffect} from 'react'; 
import SignRecognizer from './signRecognizer';


const ImageUploader = () => {
    const [image, setImage] = useState(null);
    const [classyfing, setClassyfing] = useState(null);
    const imageStyle = {
        maxWidth: '320px',
        height: '2.2vh' 
    }
    const buttonStyle = {
        marginTop: '300px', 


    }
    const classifyImage = () => {
        setClassyfing(true); 
    }
    return (
        <div>
            <h1>Upload an image</h1>
            <div>
                
                
                    <input type="file" name="image" onChange={(e) => {
                        setImage(e.target.files[0]); 
                    }}/>
            </div>
            {image && 
            (
                <div>
                    <div style={imageStyle}>
                        <img src={URL.createObjectURL(image)} alt=""/>
                    </div>
                    <div style={buttonStyle}><button onClick={classifyImage}>Classify</button></div>
                    {classyfing && <SignRecognizer image={image}/>}
                </div>
            )
            }
            
        </div>
    )




}
export default ImageUploader; 