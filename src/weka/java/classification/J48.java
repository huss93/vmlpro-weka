/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.java.classification;

import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Hussein-Badran
 */
public class J48 {
    
  /**
   * Expects an ARFF file as first argument (class attribute is assumed
   * to be the last attribute).
   *
   * @param  commandline arguments
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // load data
    
    CSVLoader loader = new CSVLoader();
    loader.setFile(new File("test.csv"));
    Instances train = loader.getStructure();
    train.setClassIndex(train.numAttributes() - 1);

    //test
    CSVLoader loaderTestingFile = new CSVLoader();
    loaderTestingFile.setFile(new File("test.csv"));
    Instances test = loaderTestingFile.getStructure();
    test.setClassIndex(test.numAttributes() - 1);

     // train classifier
    Classifier cls =  new weka.classifiers.trees.J48();
    cls.buildClassifier(train);
    System.out.println(cls);
    // evaluate classifier and print some statistics
    Evaluation eval = new Evaluation(train);
    eval.evaluateModel(cls, test);
    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    System.out.println(eval.toClassDetailsString());
  }
}
