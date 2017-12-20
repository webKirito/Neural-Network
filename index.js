
class OurData {
    constructor(input , out) {
        this.input = input;
        this.out = out;
    }
}

class Neuron{
    constructor(neuronDimension){
        this.weights = [neuronDimension];
        this.weightsLength = this.weights.length;
        
        for (let i = 0; i < neuronDimension; i++) {
            this.weights[i] = this.rnd(-0.5, 0.5);
        }
        
       // this.bias = this.rnd(0, 1);
        this.bias = 0;
        
           
    }

    rnd (min, max) {
        return Math.random() * (max - min) + min;
      }
    
    setInput(input) {
        this.input = input;
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    setWeights(weights) {
        this.weights = weights;
    }

    getWeigths() {
        return this.weights;
    }

    getBias() {
        return this.bias;
    }

    setBias(bias) {
        this.bias = bias;
    }
}


class Layer {
    
    constructor(n) {
        this.neuronArr = [n];
        for (let i = 0 ; i < n ; i++) {
            this.neuronArr[i] = new Neuron();
        }
        this.neuronsCount = this.neuronArr.length;
        this.lastInput = [];
        this.lastOutput = [];
    }

    getLastInput(){
        return this.lastInput;
    }

    setLastInput(input){
        this.lastInput = input;
    }

    getLastOutput(){
        return this.lastOutput;
    }

    setLastOutput(out){
        this.lastOutput = out;
    }

    getNeuronArr() {
        return this.neuronArr;
    }

    getNeuronsCount() {
        return this.neuronsCount;
    }

    setNeuronArr(arr) {
        this.neuronArr = arr;
    }

    getOutput(input){
        this.lastInput = input;
        let res = [];

        for (let i = 0; i < this.neuronArr.length; i++) {
            let sum = 0;
            for (let j = 0; j < input.length; j++) {
                sum += input[j] * this.neuronArr[i].weights[j];
            }
            res.push(Neuron.sigmoid(sum + this.neuronArr[i].bias));   
        }
        this.lastOutput = res;
        return res;  
    }

    getLayerError(sigma) {
        let res = [];
       
        for (let i = 0; i < this.lastInput.length; i++) {
            let sum = 0;
            for (let j = 0; j < this.neuronArr.length ; j++) {
                sum += this.neuronArr[j].weights[i] * sigma[j];
            }
            res.push(sum);   
        }

        return res;
    }

    updateWeights(sigma, speed) {
        
        let sumArr = this.getSumArr(this.lastInput);

        for (let i = 0; i < this.neuronArr.length; i++) {
            for (let j = 0; j < this.lastInput.length ; j++) {
                this.neuronArr[i].weights[j] = this.neuronArr[i].weights[j] + 
                speed * 
                sigma[i] * 
                Neuron.sigmoid(sumArr[i]) * (1 - Neuron.sigmoid(sumArr[i])) * 
                this.lastInput[j];
            }
            // this.neuronArr[i].bias = this.neuronArr[i].bias - speed * sigma[i];
       
        }
    }

    getSumArr(input) {
        let res = [];

        for (let i = 0; i < this.neuronArr.length; i++) {
            let sum = 0;
            for (let j = 0; j < input.length; j++) {
                sum += input[j] * this.neuronArr[i].weights[j];
            }
            res.push(sum);   
        }

        return res;

    }

    setWeightsCount(n) {
        
        if(this.neuronArr.length != 0) {
            for (let i = 0; i < this.neuronArr.length; i++) {
                this.neuronArr[i] = new Neuron(n);
            }
        }
        
    }
}

class NeuralWeb{

    constructor(layers) {
        this.layersArr = layers;
        this.layersArr[0].setWeightsCount(this.layersArr[0].neuronsCount);
        for (let i = 1; i < this.layersArr.length; i++) {
            this.layersArr[i].setWeightsCount(this.layersArr[i-1].neuronsCount);
        }
        this.layersCount = this.layersArr.length;
        this.lastError;
    }

    getOutput(input) {
        if (input.length != this.layersArr[0].neuronsCount) {
            alert(`Unknown input.`);
        } else {
            let nextInput = this.layersArr[0].getOutput(input);
            for (let i = 1; i < this.layersArr.length; i++) {
                nextInput = this.layersArr[i].getOutput(nextInput);
            }
            return nextInput;
        }

    }

    getError(testData, ourResult) {
        let sum = 0;
        for (let i = 0 ; i < testData.length ; i++) {
            sum += Math.pow((ourResult[i] - testData[i]),2);
        }
        return sum / 2;
    }

    teachUntilE(example , test , speed ) {
        for (let i = 0; i < example.length; i++){
            this.teach(example[i], speed);
        }
        for (let i = 0; i < test.length; i++){
            this.lastError = this.getError(this.getOutput(test[i].input),test[i].out);
        }
    }

    teach(ourData, speed){
        let arr = [];

        this.lastError = this.getError(this.getOutput(ourData.input), ourData.out);

        let y = this.getOutput(ourData.input);
        let sigma = [ourData.out.length];

        for (let i = 0; i < ourData.out.length; i++) {
            sigma[i] = (ourData.out[i] - y[i])*Neuron.sigmoid(ourData.out[i] - y[i])*(1 - Neuron.sigmoid(ourData.out[i] - y[i]));
        }

        arr.push(sigma);
        let neuronSigmas = this.layersArr[this.layersCount - 1].getLayerError(sigma);
        

        arr.push(neuronSigmas);

        for (let i = (this.layersArr.length - 2); i > 0 ; i--) {
            neuronSigmas = this.layersArr[i].getLayerError(neuronSigmas);
            arr.push(neuronSigmas);
        }
        
        arr = arr.reverse();
        

        this.layersArr[0].updateWeights(arr[0], speed);

        for (let i = 1; i < this.layersArr.length; i++)
        {
            this.layersArr[i].setLastInput(this.layersArr[i - 1].getOutput(this.layersArr[i-1].getLastInput()));
            this.layersArr[i].updateWeights(arr[i], speed);
        }
    }
}

let resArr = new Array(9);
for (let i = 0 ; i < 9 ; i ++ ){
    resArr[i] = 0;
}

for (let i = 0 ; i < 9; i++) {
    $('.grid-container').append(`<div id = 'c${i}' , class = 'cub'></div>`);
    $(`#c${i}`).click(()=> {
        
        if ($(`#c${i}`).css('background-color') == 'rgb(255, 255, 255)'){
            $(`#c${i}`).css('background-color','black');
            $(`#c${i}`).css('border-color','white');
            resArr[i] = 1;
        } else {
            $(`#c${i}`).css('background-color','white');
            $(`#c${i}`).css('border-color','black');
            resArr[i] = 0;
        }
    });
    console.log(resArr);
}



$('#test-btn').click(
    () => {
        let l1 = new Layer(2);
        let l2 = new Layer(4);
        let l3 = new Layer(1);
        let l4 = new Layer(1);
        
        let agesN = +$('#age').val();
        let speed = +$('#speed').val();

        let nn = new NeuralWeb([l1,l2,l3]);
        
        let test = nn.getOutput([0,0]);

        let res = ``;
        res += "Before training([0,0]): " + test + '\n' + '\n';
        
        let t1 = new OurData([0,0],[0]);
        let t2 = new OurData([0,1],[1]);
        let t3 = new OurData([1,0],[1]);
        let t4 = new OurData([1,1],[1]);
        
        
        for (let i = 0; i < agesN; i++)
        {
            nn.teach( t1, speed);
            nn.teach( t2, speed);
            nn.teach( t3, speed);
            nn.teach( t4, speed);
            
        }
        
        test1 = nn.getOutput([0,0]);
        test2 = nn.getOutput([0,1]);
        test3 = nn.getOutput([1,0]);
        test4 = nn.getOutput([1,1]);
        
        res += ("After trainig([0,0]): " + test1 + '\n');
        res += ("After trainig([0,1]): " + test2 + '\n');
        res += ("After training([1,0]): " + test3 + '\n');
        res += ("After training([1,1]): " + test4 + '\n' + '\n');
        res += ("Last Error: " + nn.lastError + '\n');
        
        console.log("Error on [0,0]: " + nn.getError(nn.getOutput([0,0]), [0]));
        console.log("Error on [0,1]: " + nn.getError(nn.getOutput([0,1]), [1]));
        console.log("Error on [1,0]: " + nn.getError(nn.getOutput([1,0]), [1]));
        console.log("Error on [1,1]: " + nn.getError(nn.getOutput([1,1]), [1]));

        $('#test-area').text(res);
    }
);

$('#run').click(
    () => {
        // let l1 = new Layer(9);
        // let l2 = new Layer(9);
        // let l3 = new Layer(4);

        let layerArr = [];
        let layerTxt = $('#l-n').val();
        layerTxt = layerTxt.split(" ");
        console.log(layerTxt);
        for (let i = 0; i < layerTxt.length ; i++) {
            layerArr.push(new Layer(+layerTxt[i]));
        }
        
        
        let agesN = +$('#age').val();
        let speed = +$('#speed').val();

        let nn = new NeuralWeb(layerArr);
        
        let test = loadFromFile("train.txt");

        let res = ``;
        
        
        // let t1 = new OurData([0,1,0,1,1,1,0,1,0],[0,0,0,1]);
        // let t2 = new OurData([1,1,1,1,0,1,1,1,1],[0,0,1,0]);
        // let t3 = new OurData([0,0,0,1,1,1,0,0,0],[0,1,0,0]);
        // let t4 = new OurData([1,0,1,0,1,0,1,0,1],[1,0,0,0]);
        
        
        for (let i = 0; i < agesN; i++)
        {
            for (let j = 0; j < Math.round(test.length * 0.8); j++) {
                nn.teach(test[j],speed);
            }



            // nn.teach( t1, speed);
            // nn.teach( t2, speed);
            // nn.teach( t3, speed);
            // nn.teach( t4, speed);
            
        }
        
        test1 = nn.getOutput(resArr);
        
        
        res += (`Our data([${resArr}])`+'\n'+ `Network answer ${test1}` +'\n' + "\n");

        let charSet = test1.indexOf(Math.max(...test1));;
        let char = "";
        switch (charSet) {
            case 0:
                char = "x"
                break;
            case 1:
                char = "-"
                break;
            case 2:
                char = "0"
                break;
            case 3:
                char = "+"
                break;
        }

        res += "Network has recognized '" + char + "' ."


        // res += ("After trainig([1,1,1,1,0,1,1,1,1]): " + test2 +'\n'+ 'Must be [0,0,1,0]' + '\n');
        // res += ("After training([0,0,0,1,1,1,0,0,0]): " + test3 +'\n'+ 'Must be [0,1,0,0]' + '\n');
        // res += ("After training([1,0,1,0,1,0,1,0,1]): " + test4 + '\n'+ 'Must be [1,0,0,0]' +'\n' + '\n');
        // res += ("Last Error: " + nn.lastError + '\n');
        
        // console.log("Error on [0,0]: " + nn.getError(nn.getOutput([0,0]), [0]));
        // console.log("Error on [0,1]: " + nn.getError(nn.getOutput([0,1]), [1]));
        // console.log("Error on [1,0]: " + nn.getError(nn.getOutput([1,0]), [1]));
        // console.log("Error on [1,1]: " + nn.getError(nn.getOutput([1,1]), [1]));

        $('#result').text(res);
    }
);


function loadFromFile(_url) {
    let b;

    $.ajax({
        url: _url,
        async: false,
        cache: false,
        dataType: "text",
        success: function (data, textStatus, jqXHR) {
            res = data;
        }
    });

    let dataArr = res.split("\n");
    let resArr = [];
   
    for (let i = 0; i < dataArr.length ; i++) {
        let tmp = dataArr[i].split("   ");
        resArr.push(new OurData(tmp[0].split(','), tmp[1].split(',')));    
    }
    return resArr;

}




