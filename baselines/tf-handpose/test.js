const read = require('./utilities');
const handpose = require('@tensorflow-models/handpose');
const fs = require('fs');
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

async function main() {
	var  model = await handpose.load({detectionConfidence:0, iouThreshold:0.3, scoreThreshold:0.75});
    console.log("javascript model loaded")
    console.log(process.argv);
    var p = process.argv[2];
    var sample = fs.readdirSync(p);
    //for (var i = 0; i < samples.length; ++i) {
        //var p = process.argv[2] + "/" + samples[i];
    tf.setBackend("tensorflow");
    var gt_image = await read.loadTensor(p + "/" +sample[0]);
    var pred_image = await read.loadTensor(p + "/" + sample[1]);
    tf.setBackend('cpu');
    var gt_pred = await model.estimateHands(gt_image);
    var pred_pred = await model.estimateHands(pred_image);
    fs.writeFile(p + "/" + "gt_js.json", JSON.stringify(gt_pred), 'utf8',(err) => {
                                                                if (err) throw err;
                                                                console.log('The file has been saved!');
                                                                });
    fs.writeFile(p + "/" + "pred_js.json", JSON.stringify(pred_pred), 'utf8',(err) => {
                                                                            if (err) throw err;
                                                                            console.log('The file has been saved!');
                });
   // }    
}
main();
