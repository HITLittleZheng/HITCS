
import java.io.IOException;
import java.util.regex.*;
import java.util.Iterator;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
public class WordCount {
    public WordCount() {
    }
     public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = (new GenericOptionsParser(conf, args)).getRemainingArgs();
        if(otherArgs.length < 2) {
            System.err.println("Usage: wordcount <in> [<in>...] <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(WordCount.TokenizerMapper.class);
        job.setCombinerClass(WordCount.IntSumReducer.class);
        job.setReducerClass(WordCount.IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class); 
        for(int i = 0; i < otherArgs.length - 1; ++i) {
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[otherArgs.length - 1]));
        System.exit(job.waitForCompletion(true)?0:1);
    }

     public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
    	    private static final IntWritable one = new IntWritable(1);
    	    private Text word = new Text();

    	    public TokenizerMapper() {
    	    }

    	    public void map(Object key, Text value, Mapper<Object, Text, Text, IntWritable>.Context context)
    	            throws IOException, InterruptedException {
    	        // Split DocID and the actual text
    	        String DocId = value.toString().substring(0, value.toString().indexOf("\t"));
    	        String value_raw =  value.toString().substring(value.toString().indexOf("\t") + 1);
    	        
    	        // Reading input one line at a time and tokenizing by using space, "'", and "-" characters as tokenizers.
    	        StringTokenizer itr = new StringTokenizer(value_raw, " '-");
    	        
    	        // Iterating through all the words available in that line and forming the key/value pair.
    	        while (itr.hasMoreTokens()) {
    	          // Remove special characters
    	          word.set(itr.nextToken().replaceAll("[^a-zA-Z]", ""));
    	          if(word.toString() != "" && !word.toString().isEmpty()){
    	        	  context.write(word, one);
    	          }
    	    }
    	}
     }
	public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
	        private IntWritable result = new IntWritable();
	        public IntSumReducer() {
	        }
	        public void reduce(Text key, Iterable<IntWritable> values, Reducer<Text, IntWritable, Text, IntWritable>.Context context) throws IOException, InterruptedException {
	            int sum = 0;
	            IntWritable val;
	            for(Iterator i$ = values.iterator(); i$.hasNext(); sum += val.get()) {
	                val = (IntWritable)i$.next();
	            }
	            this.result.set(sum);
	            context.write(key, this.result);
	        }
	    }
	}
     

