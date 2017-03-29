/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.java.classification;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Hussein-Badran
 */
public class Bayes {
 
    public static void main(String[] args) throws Exception {
    // load data
//    
//    ArffLoader loader = new ArffLoader();
//    loader.setFile(new File("C:\\Users\\Lenovo\\Desktop\\lurnerWork\\weka\\all-weather-data.arff"));
//    Instances train = loader.getStructure();
//    train.setClassIndex(train.numAttributes() - 1);


        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("C:\\Users\\Lenovo\\Desktop\\lurnerWork\\weka\\all-weather-data.arff");
        Instances train = source1.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("C:\\Users\\Lenovo\\Desktop\\lurnerWork\\weka\\play-testing-set.arff");
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // model

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);


        // this does the trick  
        double label = naiveBayes.classifyInstance(test.instance(0));//zero if class yes and 1 if class no
        test.instance(0).setClassValue(label);
        System.out.println("the label number is : "+label);
        System.out.println("The Prediction Class is : "+test.instance(0).stringValue(4));
  }
    
}
