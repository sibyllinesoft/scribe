const path = require("path");
const fs = require("fs");

/**
 * Get the platform-specific package name
 */
function getPlatformPackage() {
  const platform = process.platform;
  const arch = process.arch;

  const platformMap = {
    "darwin-arm64": "@sibyllinesoft/scribe-darwin-arm64",
    "darwin-x64": "@sibyllinesoft/scribe-darwin-x64",
    "linux-arm64": "@sibyllinesoft/scribe-linux-arm64",
    "linux-x64": "@sibyllinesoft/scribe-linux-x64",
    "win32-x64": "@sibyllinesoft/scribe-win32-x64",
  };

  const key = `${platform}-${arch}`;
  const pkg = platformMap[key];

  if (!pkg) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. ` +
        `Supported platforms: ${Object.keys(platformMap).join(", ")}`
    );
  }

  return pkg;
}

/**
 * Get the path to the scribe binary
 */
function getBinaryPath() {
  const pkg = getPlatformPackage();

  // Try to resolve the platform-specific package
  try {
    const pkgPath = require.resolve(`${pkg}/package.json`);
    const pkgDir = path.dirname(pkgPath);
    const binaryName = process.platform === "win32" ? "scribe.exe" : "scribe";
    const binaryPath = path.join(pkgDir, "bin", binaryName);

    if (fs.existsSync(binaryPath)) {
      return binaryPath;
    }

    throw new Error(`Binary not found at ${binaryPath}`);
  } catch (err) {
    throw new Error(
      `Failed to find scribe binary. Package ${pkg} may not be installed.\n` +
        `Try reinstalling with: npm install @sibyllinesoft/scribe\n` +
        `Original error: ${err.message}`
    );
  }
}

module.exports = {
  getBinaryPath,
  getPlatformPackage,
};
