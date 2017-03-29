/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.java.classification;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesUpdateable;

import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

/**
 * This example trains NaiveBayes incrementally on data obtained
 * from the ArffLoader.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class IncrementalClassifier {

  /**
   * Expects an ARFF file as first argument (class attribute is assumed
   * to be the last attribute).
   *
   * @param  commandline arguments
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // load data
    
    ArffLoader loader = new ArffLoader();
    loader.setFile(new File("C:\\Users\\Lenovo\\Desktop\\lurnerWork\\weka\\all-weather-data.arff"));
    Instances structure = loader.getStructure();
    structure.setClassIndex(structure.numAttributes() - 1);

    ArffLoader loader2 = new ArffLoader();
    loader2.setFile(new File("C:\\Users\\Lenovo\\Desktop\\lurnerWork\\weka\\play-testing-set.arff"));
    Instances test = loader2.getStructure();
    test.setClassIndex(structure.numAttributes() - 1);
    
    
    // train NaiveBayes
    NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
    nb.buildClassifier(structure);
    Instance current;
    while ((current = loader.getNextInstance(structure)) != null)
      nb.updateClassifier(current);

    // output generated model
    System.out.println(nb);
    
    Classifier cModel = (Classifier)new NaiveBayes();
    cModel.buildClassifier(structure);
    Evaluation eTest = new Evaluation(structure);
    eTest.evaluateModel(cModel, test);
    String strSummary = eTest.toSummaryString();
    System.out.println(strSummary);
    System.out.println(eTest.toClassDetailsString());
    double[][] cmMatrix = eTest.confusionMatrix();
    for(int i=0;i<cmMatrix.length;i++)
    {
        for(int j=0;j<cmMatrix.length;j++)
        {
              System.out.println(cmMatrix[i][j]);
        }
        
    }
    
    
    
    //System.out.println(test.instance(0).toString());
  }
}