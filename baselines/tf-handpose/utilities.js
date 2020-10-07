const fs = require('fs');
const jpeg = require("jpeg-js");
const tf = require('@tensorflow/tfjs')
const node = require('@tensorflow/tfjs-node')


function readImage(path) {	
  var buf = fs.readFileSync(path);
  var pixels = jpeg.decode(buf, true);
  return pixels;
}

module.exports= {
	loadTensor: function(path, channel) {
		var buf = fs.readFileSync(path);
		var tensor = node.node.decodeImage(buf, channel);
		tensor = tf.image.resizeNearestNeighbor(tensor, [256, 256]);
		return tensor;
	},
};

