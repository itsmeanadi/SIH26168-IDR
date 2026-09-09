package com.idr.app;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private WebView webView;
    private SensorBridge sensorBridge;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private final String LOCAL_URL = "http://10.89.225.17:8000";
    private final String RENDER_URL = "https://sih26168-idr-3.onrender.com";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webView);
        configureWebView();

        sensorBridge = new SensorBridge(this, webView);

        requestPermissions();

        // Default immediately to Render or Local while checking in background
        webView.loadUrl(RENDER_URL);

        // Check backend availability asynchronously without blocking the UI thread
        determineBackendAsync(url -> {
            runOnUiThread(() -> {
                if (webView != null && !url.equals(RENDER_URL)) {
                    webView.loadUrl(url);
                }
            });
        });
    }

    interface BackendCallback {
        void onResult(String url);
    }

    private void determineBackendAsync(BackendCallback callback) {
        executor.execute(() -> {
            if (isBackendAvailable(LOCAL_URL)) {
                callback.onResult(LOCAL_URL);
            } else if (isBackendAvailable(RENDER_URL)) {
                callback.onResult(RENDER_URL);
            } else {
                callback.onResult(RENDER_URL);
            }
        });
    }

    private boolean isBackendAvailable(String urlString) {
        try {
            URL url = new URL(urlString + "/api/system/health");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(2000);
            connection.setReadTimeout(2000);
            connection.setRequestMethod("GET");
            int responseCode = connection.getResponseCode();
            connection.disconnect();
            return (responseCode == HttpURLConnection.HTTP_OK);
        } catch (IOException e) {
            return false;
        }
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        webView.setWebViewClient(new WebViewClient());
    }

    private void requestPermissions() {
        String[] permissions = {
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, permissions, 100);
        } else {
            if (sensorBridge != null) {
                sensorBridge.start();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 100 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (sensorBridge != null) {
                sensorBridge.start();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorBridge != null) {
            sensorBridge.start();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorBridge != null) {
            sensorBridge.stop();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (sensorBridge != null) {
            sensorBridge.stop();
        }
        executor.shutdown();
    }
}
