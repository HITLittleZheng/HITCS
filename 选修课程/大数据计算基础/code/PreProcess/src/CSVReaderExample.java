import com.opencsv.CSVReader;

import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Arrays;

public class CSVReaderExample {

    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java CSVReaderExample <folderPath>");
            return;
        }

        String folderPath = args[0];
        File folder = new File(folderPath);

        if (!folder.isDirectory()) {
            System.out.println("Provided path is not a directory.");
            return;
        }

        File outputFolder = new File(folder, "output");
        if (!outputFolder.exists()) {
            if (!outputFolder.mkdir()) {
                System.out.println("Failed to create output folder.");
                return;
            }
        }

        File[] csvFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));

        if (csvFiles == null || csvFiles.length == 0) {
            System.out.println("No CSV files found in the specified directory.");
            return;
        }

        for (File csvFile : csvFiles) {
            processCSVFile(csvFile, outputFolder, "articleID", "snippet");
        }
    }

    private static int getColumnIndex(String[] headers, String columnName) {
        return Arrays.asList(headers).indexOf(columnName);
    }

    private static void processCSVFile(File csvFile, File outputFolder, String articleIDColumnName, String snippetColumnName) {
        try (CSVReader csvReader = new CSVReaderBuilder(new FileReader(csvFile)).withSkipLines(0).build();
             FileWriter writer = new FileWriter(new File(outputFolder, csvFile.getName().replace(".csv", "_output.txt")))) {

            List<String[]> records = csvReader.readAll();
            
            // Find the column indices dynamically based on column names
            int articleIDIndex = getColumnIndex(records.get(0), articleIDColumnName);
            int snippetIndex = getColumnIndex(records.get(0), snippetColumnName);
            boolean flag=false;
            
            for (String[] record : records) {
                if (record.length > Math.max(articleIDIndex, snippetIndex) && flag) {
                    String articleID = record[articleIDIndex];
                    String snippet = record[snippetIndex].toLowerCase();

                    writer.write(articleID + "\t" + snippet + "\n");
                }
                flag=true;
            }

            System.out.println("Data has been written to " +
                    Paths.get(outputFolder.getAbsolutePath(), csvFile.getName().replace(".csv", "_output.txt")));

        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
    }
}
