package com.idr.app;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private WebView webView;
    private SensorBridge sensorBridge;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private static final String PREFS_NAME = "IDR_PREFERENCES";
    private static final String KEY_CUSTOM_URL = "custom_backend_url";
    private static final String RENDER_URL = "https://sih26168-idr-3.onrender.com";
    private static final String EMULATOR_URL = "http://10.0.2.2:8000";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Check if intent contains backend override
        handleIntentOverrides(getIntent());

        webView = findViewById(R.id.webView);
        configureWebView();

        sensorBridge = new SensorBridge(this, webView);

        requestPermissions();

        // Load saved URL or fallback to Render immediately while probing in background
        String initialUrl = getSavedBackendUrl();
        webView.loadUrl(initialUrl != null ? initialUrl : RENDER_URL);

        // Check backend candidates asynchronously without blocking UI thread
        determineBackendAsync(url -> {
            runOnUiThread(() -> {
                if (webView != null && (initialUrl == null || !initialUrl.equals(url))) {
                    saveBackendUrl(url);
                    webView.loadUrl(url);
                }
            });
        });
    }

    private void handleIntentOverrides(Intent intent) {
        if (intent != null) {
            String overrideUrl = intent.getStringExtra("BACKEND_URL");
            if (overrideUrl == null) overrideUrl = intent.getStringExtra("SERVER_URL");
            if (overrideUrl != null && !overrideUrl.trim().isEmpty()) {
                saveBackendUrl(overrideUrl.trim());
            }
        }
    }

    private String getSavedBackendUrl() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return prefs.getString(KEY_CUSTOM_URL, null);
    }

    private void saveBackendUrl(String url) {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().putString(KEY_CUSTOM_URL, url).apply();
    }

    interface BackendCallback {
        void onResult(String url);
    }

    private void determineBackendAsync(BackendCallback callback) {
        executor.execute(() -> {
            List<String> candidates = new ArrayList<>();
            String savedUrl = getSavedBackendUrl();
            if (savedUrl != null && !savedUrl.trim().isEmpty() && !candidates.contains(savedUrl.trim())) {
                candidates.add(savedUrl.trim());
            }
            if (!candidates.contains(EMULATOR_URL)) {
                candidates.add(EMULATOR_URL);
            }
            if (!candidates.contains(RENDER_URL)) {
                candidates.add(RENDER_URL);
            }

            for (String candidate : candidates) {
                if (isBackendAvailable(candidate)) {
                    callback.onResult(candidate);
                    return;
                }
            }
            callback.onResult(RENDER_URL);
        });
    }

    private boolean isBackendAvailable(String urlString) {
        try {
            URL url = new URL(urlString + "/api/system/health");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(1500);
            connection.setReadTimeout(1500);
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

        // Expose bridge for runtime backend URL configuration & settings from WebView
        webView.addJavascriptInterface(new Object() {
            @JavascriptInterface
            public void setBackendUrl(String url) {
                if (url != null && !url.trim().isEmpty()) {
                    saveBackendUrl(url.trim());
                    runOnUiThread(() -> {
                        if (webView != null) webView.loadUrl(url.trim());
                    });
                }
            }

            @JavascriptInterface
            public String getBackendUrl() {
                return getSavedBackendUrl();
            }

            @JavascriptInterface
            public void openLocationSettings() {
                try {
                    Intent intent = new Intent(android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                    intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(intent);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "AndroidConfig");

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
