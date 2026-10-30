plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.mobiinfer.demo"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }
    ndkVersion = "27.0.12077973"

    defaultConfig {
        // `namespace` stays as the Kotlin package (it only names the generated R
        // class). `applicationId` is the real app identity seen by the device and
        // Play Store, so it must not be the `com.example.*` scaffold default.
        applicationId = "com.mobiinfer.demo"
        // minSdk = 24 (Android 7.0) covers ~99% of in-use devices.
        // The native code only requires arm64-v8a (Android 5.0+), and the
        // app itself uses no Android 13+ APIs. The adaptive icon XML is
        // placed under mipmap-anydpi-v26/ so pre-Oreo devices fall back
        // to the WebP icons in mipmap-{m,h,xh,xxh,xxxh}dpi/.
        minSdk = 24
        targetSdk = 36
        versionCode = 4
        versionName = "0.3.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters.add("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DLLAMA_BUILD_TOOLS=ON"
                arguments += "-DLLAMA_OPENSSL=OFF"

                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=ON"
                arguments += "-DLLAMA_CURL=OFF"

                // Qualcomm Hexagon (HTP/NPU) backend. The hexagon reg is only
                // registered when ggml-base is compiled with GGML_HEXAGON=ON;
                // a prebuilt libggml-hexagon.so alone is never picked up.
                // PREBUILT_LIB_DIR must be non-empty or the SDK's
                // hexagon_fun.cmake string(FIND) aborts.
                val hexagonSdk = System.getenv("HEXAGON_SDK_ROOT")
                    ?: error("Set HEXAGON_SDK_ROOT to the Hexagon SDK installation directory")
                val hexagonTools = System.getenv("HEXAGON_TOOLS_ROOT")
                    ?: "$hexagonSdk/tools/HEXAGON_Tools/19.0.07"
                arguments += "-DGGML_HEXAGON=ON"
                arguments += "-DPREBUILT_LIB_DIR=toolv19_v79"
                arguments += "-DHEXAGON_SDK_ROOT=$hexagonSdk"
                arguments += "-DHEXAGON_TOOLS_ROOT=$hexagonTools"

                // Offline/CI escape hatch: reuse a pre-populated KleidiAI checkout
                // instead of letting FetchContent hit the network.
                System.getenv("KLEIDIAI_SRC")?.let {
                    arguments += "-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD=$it"
                }
            }
        }
    }

    // Release signing config. Credentials live in ~/.gradle/gradle.properties
    // (outside any git repo, only readable by your local mac account).
    // If those properties aren't set the release build still works but produces
    // an unsigned apk - useful e.g. on CI without secrets.
    signingConfigs {
        create("release") {
            val keystorePath = providers.gradleProperty("MINICPMV_KEYSTORE").orNull
            if (!keystorePath.isNullOrBlank()) {
                storeFile = file(keystorePath)
                storePassword = providers.gradleProperty("MINICPMV_KEYSTORE_PASSWORD").orNull
                keyAlias = providers.gradleProperty("MINICPMV_KEY_ALIAS").orNull
                keyPassword = providers.gradleProperty("MINICPMV_KEY_PASSWORD").orNull
            }
        }
    }

    buildTypes {
        release {
            // Keep ProGuard/R8 disabled: the app calls native JNI symbols and
            // shrinking the Kotlin side has no measurable benefit here, while
            // an over-aggressive shrinker is the most common cause of crashes
            // in apps with lots of JNI bindings.
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Only attach the signing config when the keystore actually exists,
            // so contributors without the secret can still run :assembleRelease
            // (it'll produce an unsigned apk in that case).
            val signingCfg = signingConfigs.getByName("release")
            if (signingCfg.storeFile?.exists() == true) {
                signingConfig = signingCfg
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }
    buildFeatures {
        viewBinding = true
    }

    androidResources {
        noCompress.add("gguf")
        noCompress.add("bin")
    }

    packaging {
        jniLibs {
            // The HTP skel is a Qualcomm DSP6 ELF loaded by the DSP at runtime;
            // the NDK strip step must not touch it.
            keepDebugSymbols.add("**/libggml-htp-*.so")
            // FastRPC resolves the skel as a real file on disk (ADSP_LIBRARY_PATH
            // -> nativeLibraryDir). With extractNativeLibs=false the libs only
            // exist inside the APK and the DSP loader fails with
            // AEE_EUNABLETOLOAD (0x80000406). Force extraction.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // Markdown rendering for AI streaming responses (headings, bold, lists, code, etc.)
    implementation("io.noties.markwon:core:4.6.2")
    implementation("io.noties.markwon:syntax-highlight:4.6.2") {
        // Kotlin already supplies these annotations under the newer artifact.
        exclude(group = "org.jetbrains", module = "annotations-java5")
    }
    annotationProcessor("io.noties:prism4j-bundler:2.0.0")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

// ---------------------------------------------------------------------------
// Dynamic CPU dispatch: build an additional libggml-cpu.so optimised for
// ARMv8.6-a (i8mm + bf16) and package it alongside the baseline build.
// At runtime the Kotlin CpuFeatures helper detects hardware capabilities
// and pre-loads the best variant before the rest of the native chain.
// ---------------------------------------------------------------------------

fun runCmd(vararg args: String) {
    val proc = ProcessBuilder(*args).inheritIO().start()
    val rc = proc.waitFor()
    if (rc != 0) error("Command failed (rc=$rc): ${args.joinToString(" ")}")
}

val sdkRoot: String = System.getenv("ANDROID_HOME")
    ?: file("../local.properties").takeIf { it.exists() }?.readLines()
        ?.firstOrNull { it.startsWith("sdk.dir=") }?.substringAfter("=")
    ?: error("Cannot locate Android SDK — set ANDROID_HOME or local.properties")

tasks.register("buildGgmlCpu_v86") {
    group = "native"
    description = "Build libggml-cpu optimised for armv8.6-a+i8mm+bf16"

    val destSo = file("src/main/jniLibs/arm64-v8a/libggml-cpu-v86.so")
    outputs.file(destSo)

    doLast {
        val cmake = "$sdkRoot/cmake/4.1.2/bin/cmake"
        val toolchain = "$sdkRoot/ndk/27.0.12077973/build/cmake/android.toolchain.cmake"
        val bd = File(project.layout.buildDirectory.asFile.get(), "v86-cmake/arm64-v8a")
        bd.mkdirs()

        runCmd(
            cmake,
            "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_PLATFORM=android-24",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DGGML_NATIVE=OFF",
            "-DGGML_LLAMAFILE=ON",
            "-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16",
            "-S", file("src/main/cpp").absolutePath,
            "-B", bd.absolutePath,
        )

        runCmd(
            cmake,
            "--build", bd.absolutePath,
            "--target", "ggml-cpu",
            "-j", Runtime.getRuntime().availableProcessors().toString(),
        )

        val builtSo = fileTree(bd).matching { include("**/libggml-cpu.so") }.singleFile
        destSo.parentFile.mkdirs()
        builtSo.copyTo(destSo, overwrite = true)
        logger.lifecycle("Copied v86 ggml-cpu -> ${destSo.absolutePath} (${destSo.length() / 1024}K)")
    }
}

// ---------------------------------------------------------------------------
// Hexagon DSP skels. The ggml-hexagon CMakeLists declares the HTP skels as
// ExternalProject targets and only installs them via `cmake --install`, which
// the Android Gradle Plugin never runs. So `:app:assemble*` alone produces an
// APK with the host-side libggml-hexagon.so but NO libggml-htp-v*.so, and the
// DSP fails to open a session at load time (AEE_EUNABLETOLOAD 0x80000406).
// Build the skels here and drop them into jniLibs so they ship in the APK.
// Override versions with -PhtpSkels=v79 (comma separated) to trim build time.
// ---------------------------------------------------------------------------

val htpSkels: List<String> =
    (project.findProperty("htpSkels") as String?)?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }
        ?: listOf("v73", "v75", "v79", "v81")

tasks.register("buildHtpSkels") {
    group = "native"
    description = "Build libggml-htp-v*.so DSP skels and stage them in jniLibs"

    val hexagonSdk = System.getenv("HEXAGON_SDK_ROOT")
        ?: error("Set HEXAGON_SDK_ROOT to the Hexagon SDK installation directory")
    val hexagonTools = System.getenv("HEXAGON_TOOLS_ROOT")
        ?: File(hexagonSdk, "tools/HEXAGON_Tools")
            .listFiles { file -> file.isDirectory }
            ?.maxByOrNull { it.name }
            ?.absolutePath
        ?: error("Set HEXAGON_TOOLS_ROOT to the Hexagon tools directory")

    val htpSrc = file("../../../ggml/src/ggml-hexagon/htp").canonicalFile
    val destDir = file("src/main/jniLibs/arm64-v8a")
    val outputsList = htpSkels.map { destDir.resolve("libggml-htp-$it.so") }

    inputs.dir(htpSrc)
    inputs.property("htpSkels", htpSkels)
    inputs.property("hexagonSdk", File(hexagonSdk).canonicalPath)
    inputs.property("hexagonTools", File(hexagonTools).canonicalPath)
    outputs.files(outputsList)

    doLast {
        val cmake = "$sdkRoot/cmake/4.1.2/bin/cmake"
        destDir.mkdirs()
        val selected = htpSkels.map { "libggml-htp-$it.so" }.toSet()
        destDir.listFiles { file ->
            file.isFile && file.name.startsWith("libggml-htp-") &&
                file.extension == "so" && file.name !in selected
        }?.forEach { stale ->
            stale.delete()
            logger.lifecycle("Removed stale HTP skel ${stale.absolutePath}")
        }
        for (ver in htpSkels) {
            val bd = File(project.layout.buildDirectory.asFile.get(), "htp-cmake/$ver")
            bd.mkdirs()
            runCmd(
                cmake,
                "-S", htpSrc.absolutePath,
                "-B", bd.absolutePath,
                "-DCMAKE_TOOLCHAIN_FILE=${htpSrc.absolutePath}/cmake-toolchain.cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDSP_VERSION=$ver",
                "-DPREBUILT_LIB_DIR=toolv19_$ver",
                "-DHEXAGON_SDK_ROOT=$hexagonSdk",
                "-DHEXAGON_TOOLS_ROOT=$hexagonTools",
            )
            runCmd(
                cmake,
                "--build", bd.absolutePath,
                "-j", Runtime.getRuntime().availableProcessors().toString(),
            )
            val builtSo = fileTree(bd).matching { include("**/libggml-htp-$ver.so") }.singleFile
            val destSo = destDir.resolve("libggml-htp-$ver.so")
            builtSo.copyTo(destSo, overwrite = true)
            logger.lifecycle("Copied $ver HTP skel -> ${destSo.absolutePath} (${destSo.length() / 1024}K)")
        }
    }
}

// ---------------------------------------------------------------------------
// Text-to-image: build the sdcpp-hexagon fork of stable-diffusion.cpp +
// the imagegen JNI wrapper as a self-contained set of shared libraries and
// bundle them into jniLibs.
//
// sdcpp-hexagon vendors its own (patched) ggml WITH the ggml-hexagon backend,
// whose CMake targets collide with the llama.cpp-omni ggml. It is therefore
// built as a separate CMake project (tools/imagegen) producing
// libimagegen.so plus its own libstable-diffusion.so (ggml inside) and the
// libggml-htp-v*.so skels, loaded independently at runtime.
//
// Requires HEXAGON_SDK_ROOT / HEXAGON_TOOLS_ROOT (same as the llama hexagon
// build). Skels are built for the DSP versions we ship (v79 = SM8750, v81 =
// SM8850); the runtime picks the matching one via GGML_HEXAGON_ARCH.
// ---------------------------------------------------------------------------
val buildImageGen = tasks.register("buildImageGen") {
    group = "native"
    description = "Build sdcpp-hexagon + imagegen shared libs for arm64-v8a"

    val destDir = file("src/main/jniLibs/arm64-v8a")
    val repoRoot = file("../../..").absoluteFile
    val srcDir = File(repoRoot, "tools/imagegen")
    val sdSrcDir = File(repoRoot, "vendor/sdcpp-hexagon")
    outputs.dir(destDir)

    doLast {
        if (!File(sdSrcDir, "CMakeLists.txt").exists()) {
            logger.lifecycle("Skipping buildImageGen: vendor/sdcpp-hexagon not present")
            return@doLast
        }
        val hexagonSdk = System.getenv("HEXAGON_SDK_ROOT")
            ?: error("Set HEXAGON_SDK_ROOT to the Hexagon SDK installation directory")
        val hexagonTools = System.getenv("HEXAGON_TOOLS_ROOT")
            ?: "$hexagonSdk/tools/HEXAGON_Tools/21.0.04"

        val cmake = "$sdkRoot/cmake/4.1.2/bin/cmake"
        val toolchain = "$sdkRoot/ndk/27.0.12077973/build/cmake/android.toolchain.cmake"
        val ndkLibs = "$sdkRoot/ndk/27.0.12077973/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/aarch64"
        val bd = File(project.layout.buildDirectory.asFile.get(), "imagegen-cmake/arm64-v8a")
        bd.mkdirs()

        runCmd(
            cmake,
            "-S", srcDir.absolutePath,
            "-B", bd.absolutePath,
            "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_PLATFORM=android-29",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSDCPP_SOURCE_DIR=${sdSrcDir.absolutePath}",
            "-DHEXAGON_SDK_ROOT=$hexagonSdk",
            "-DHEXAGON_TOOLS_ROOT=$hexagonTools",
            // SDK 6.7's hexagon_fun.cmake breaks on an empty PREBUILT_LIB_DIR;
            // any non-empty value works for the top-level configure, the skels
            // set their own per-DSP value.
            "-DPREBUILT_LIB_DIR=toolv19_v79",
        )
        // Single-job build: stable-diffusion.cpp translation units are very
        // memory-heavy (each clang peak ~0.8-1.5 GiB) and OOM a 16 GiB host
        // when several run at once. Override with IMAGEGEN_JOBS if desired.
        val jobs = System.getenv("IMAGEGEN_JOBS") ?: "1"
        runCmd(cmake, "--build", bd.absolutePath, "--target", "imagegen", "htp-v79", "htp-v81", "-j", jobs)

        destDir.mkdirs()
        val soNames = listOf("libimagegen.so", "libstable-diffusion.so")
        for (name in soNames) {
            val f = fileTree(bd).matching { include("**/$name") }.files.firstOrNull()
                ?: error("buildImageGen: missing $name under ${bd.absolutePath}")
            f.copyTo(File(destDir, name), overwrite = true)
            logger.lifecycle("Copied $name -> ${File(destDir, name).absolutePath}")
        }
        // HTP skels.
        for (name in listOf("libggml-htp-v79.so", "libggml-htp-v81.so")) {
            val f = fileTree(bd).matching { include("**/ggml-hexagon/$name") }.files.firstOrNull()
                ?: error("buildImageGen: missing skel $name under ${bd.absolutePath}")
            f.copyTo(File(destDir, name), overwrite = true)
            logger.lifecycle("Copied $name -> ${File(destDir, name).absolutePath}")
        }
        // Runtime dep: OpenMP (libstable-diffusion.so NEEDED). The wrapper and
        // sd lib use the NDK's static libc++ (no libc++_shared dependency), so
        // we deliberately do NOT stage libc++_shared.so - it would only risk
        // clashing with the main app's runtime.
        File("$ndkLibs/libomp.so").takeIf { it.exists() }?.let {
            it.copyTo(File(destDir, "libomp.so"), overwrite = true)
            logger.lifecycle("Copied libomp.so from NDK")
        }
    }
}

afterEvaluate {
    listOf("Debug", "Release").forEach { buildType ->
        tasks.findByName("merge${buildType}JniLibFolders")?.dependsOn("buildGgmlCpu_v86")
        tasks.findByName("merge${buildType}JniLibFolders")?.dependsOn("buildHtpSkels")
        tasks.findByName("merge${buildType}JniLibFolders")?.dependsOn(buildImageGen)
    }
}
