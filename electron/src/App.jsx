import { Component } from 'react';
import io from './utils/io';
import * as tf from '@tensorflow/tfjs';

export default class App extends Component {

	componentDidMount() {
	    // init SignaturePad
	    this.drawElement = document.getElementById('draw-area');
	    this.signaturePad = new SignaturePad(this.drawElement, {
	       minWidth: 6,
	       maxWidth: 6,
	       penColor: 'white',
	       backgroundColor: 'black',
	    });

	    // load pre-trained model
	    tf.loadModel('./model/model.json')
	      .then(pretrainedModel => {
	        document.getElementById('predict-button').classList.remove('is-loading');
	        this.model = pretrainedModel;
	      });
	}

    getImageData() {
      const inputWidth = 28;
      const inputHeight = 28;

      // resize
      const tmpCanvas = document.createElement('canvas').getContext('2d');
      tmpCanvas.drawImage(this.drawElement, 0, 0, inputWidth, inputHeight);

      // convert grayscale
      let imageData = tmpCanvas.getImageData(0, 0, inputWidth, inputHeight);
      for (let i = 0; i < imageData.data.length; i+=4) {
        const avg = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
        imageData.data[i] = imageData.data[i+1] = imageData.data[i+2] = avg;
      }

      return imageData;
    }

    getAccuracyScores(imageData) {

      const score = tf.tidy(() => {
        // convert to tensor (shape: [width, height, channels])  
        const channels = 1; // grayscale              
        let input = tf.fromPixels(imageData, channels);

        // normalized
        input = tf.cast(input, 'float32').div(tf.scalar(255));

        // reshape input format (shape: [batch_size, width, height, channels])
        input = input.expandDims();

        // predict
        return this.model.predict(input).dataSync();
      });

      return score;
    }

    prediction() {
      const imageData = this.getImageData();
      const accuracyScores = this.getAccuracyScores(imageData);
      const maxAccuracy = accuracyScores.indexOf(Math.max.apply(null, accuracyScores));

//      console.log(maxAccuracy);
	  io.emit('dispatch', maxAccuracy);
    }

    reset() {
      this.signaturePad.clear();
    }


  render() {
    return (
	  <div className="container" style={{margin:'20px'}}>
	    <div className="title" style={{fontSize:'24px'}}>Node.js for Max8 with TensorFlow.js</div>
	    <div className="columns is-centered">
	      <div className="column is-narrow">
	        <canvas id="draw-area" width="280" height="280" style={{border: '2px solid'}}></canvas>
	        <div className="field is-grouped">
	          <p className="control">
	            <a id="predict-button" className="button is-link is-loading" onClick={()=>{this.prediction();}}>
	              Prediction
	            </a>
	          </p>
	          <p className="control">
	            <a className="button" onClick={()=>{this.reset();}}>
	              Reset
	            </a>
	          </p>
	        </div>
	      </div>
	    </div>
	  </div>
    );
  }
}

