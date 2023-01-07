#!/usr/bin/env -S deno run --allow-net --allow-read --allow-write

import {
  Command,
  EnumType,
} from "https://deno.land/x/cliffy@v0.25.7/command/mod.ts";
import { tgz } from "https://deno.land/x/compress@v0.4.5/mod.ts";
import { Octokit } from "https://cdn.skypack.dev/@octokit/rest?dts";
import { basename, join } from "https://deno.land/std@0.171.0/path/mod.ts";
import { arch } from "https://deno.land/std@0.170.0/node/process.ts";
import {
  Uint8ArrayReader,
  Uint8ArrayWriter,
  ZipReader,
} from "https://deno.land/x/zipjs@v2.6.61/index.js";

const DEFAULT_OUTPUT = "./voicevox_core";

const ORGANIZATION_NAME = "VOICEVOX";

const CORE_DISPLAY_NAME = "voicevox_core";
const CORE_REPO_NAME = "voicevox_core";

const ADDITIONAL_LIBRARIES_DISPLAY_NAME = "voicevox_additional_libraries";
const ADDITIONAL_LIBRARIES_REPO_NAME = "voicevox_additional_libraries";

const OPEN_JTALK_DIC_DISPLAY_NAME = "open_jtalk_dic";
const OPEN_JTALK_DIC_URL = new URL(
  "https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz",
);

async function main(): Promise<void> {
  // CliffyはASCII文字のことしか考えていないらしく、全角文字を入れると
  // helpの表示が崩れる
  const { options } = await new Command()
    .name("download")
    .description(`Download ${CORE_DISPLAY_NAME} and other libraries.`)
    .type("accelerator", new EnumType(["cpu", "cuda", "directml"]))
    .type("cpu-arch", new EnumType(["x64", "aarch64"]))
    .type("os", new EnumType(["windows", "linux", "osx"]))
    .option("--min", `Only Download ${CORE_DISPLAY_NAME}.`)
    .option(
      "-o, --output <output>",
      "Output to the directory.",
      { default: DEFAULT_OUTPUT },
    )
    .option(
      "-v, --version <tag-or-latest>",
      `Version of ${CORE_DISPLAY_NAME}.`,
      { default: "latest" },
    )
    .option(
      "--additional-libraries-version <tag-or-latest>",
      "Version of the additional libraries.",
      { default: "latest" },
    )
    .option(
      "--accelerator <accelerator:accelerator>",
      "Accelerator. (cuda is only available on Linux)",
      { default: "cpu" },
    )
    .option(
      "--cpu-arch <cpu-arch:cpu-arch>",
      "CPU Architecture. Defaults to the current one.",
      { default: defaultArch() },
    )
    .option(
      "--os <os:os>",
      "OS. Defaults to the current one.",
      { default: defaultOS() },
    )
    .parse(Deno.args);

  if (!options.cpuArch) {
    throw new Error(`${arch}はサポートされていない環境です`);
  }

  const { output, version, additionalLibrariesVersion } = options;
  const min = !!options.min;
  const accelerator = options.accelerator as "cpu" | "cuda" | "directml";
  const cpuArch = options.cpuArch as "x64" | "aarch64";
  const os = options.os as "windows" | "linux" | "osx";

  const octokit = new Octokit();

  const coreAsset = await findGHAsset(
    octokit,
    CORE_REPO_NAME,
    version,
    (tag) => {
      const cpuArchRename = os == "linux" && cpuArch == "aarch64"
        ? "arm64"
        : cpuArch;
      const acceleratorRename = os == "linux" && accelerator == "cuda"
        ? "gpu"
        : accelerator;
      return `${CORE_DISPLAY_NAME}-${os}-${cpuArchRename}-` +
        `${acceleratorRename}-${tag}.zip`;
    },
  );

  const additionalLibrariesAsset = accelerator == "cpu"
    ? undefined
    : await findGHAsset(
      octokit,
      ADDITIONAL_LIBRARIES_REPO_NAME,
      additionalLibrariesVersion,
      (_) => {
        const acceleratorRename = accelerator == "cuda" ? "CUDA" : "DirectML";
        return `${acceleratorRename}-${os}-${cpuArch}.zip`;
      },
    );

  info(`対象OS: ${os}`);
  info(`対象CPUアーキテクチャ: ${cpuArch}`);
  info(`ダウンロードアーティファクトタイプ: ${accelerator}`);
  info(`ダウンロード${CORE_DISPLAY_NAME}バージョン: ${coreAsset.releaseTag}`);
  if (additionalLibrariesAsset) {
    info(
      `ダウンロード追加ライブラリバージョン: ` +
        `${additionalLibrariesAsset.releaseTag}`,
    );
  }

  const promises = [download(
    CORE_DISPLAY_NAME,
    coreAsset.url,
    { accept: "application/octet-stream", format: "zip", junkPaths: true },
    output,
  )];

  if (!min) {
    promises.push(download(
      OPEN_JTALK_DIC_DISPLAY_NAME,
      OPEN_JTALK_DIC_URL,
      { accept: "application/x-gzip", format: "tgz", junkPaths: false },
      output,
    ));

    if (additionalLibrariesAsset) {
      promises.push(download(
        ADDITIONAL_LIBRARIES_DISPLAY_NAME,
        additionalLibrariesAsset.url,
        { accept: "application/octet-stream", format: "zip", junkPaths: true },
        output,
      ));
    }
  }

  await Promise.all(promises);

  success("全ての必要なファイルダウンロードが完了しました");
}

function defaultArch(): "x64" | "aarch64" | undefined {
  switch (arch) {
    case "x64":
      return "x64";
    case "arm64":
      return "aarch64";
    default:
      return undefined;
  }
}

function defaultOS(): "windows" | "linux" | "osx" {
  if (Deno.build.os == "darwin") {
    return "osx";
  }
  return Deno.build.os;
}

async function findGHAsset(
  octokit: Octokit,
  repo: string,
  gitTagOrLatest: string,
  assetName: (tag: string) => string,
): Promise<{ releaseTag: string; url: URL }> {
  // FIXME: どうにかして型付けできないか?
  const endpoint = gitTagOrLatest == "latest"
    ? `GET /repos/${ORGANIZATION_NAME}/${repo}/releases/latest`
    : `GET /repos/${ORGANIZATION_NAME}/${repo}/releases/tags/${gitTagOrLatest}`;
  const { data: { html_url, tag_name, assets } } = await octokit.request(
    endpoint,
  );
  const targetAssetName = assetName(tag_name);
  const asset = assets.find((a: { name: string }) => a.name == targetAssetName);
  if (!asset) {
    throw new Error(`Could not find ${targetAssetName} in ${html_url}`);
  }
  return { releaseTag: tag_name, url: new URL(asset.url) };
}

async function download(
  displayName: string,
  url: URL,
  kind:
    | { accept: "application/octet-stream"; format: "zip"; junkPaths: true }
    | { accept: "application/x-gzip"; format: "tgz"; junkPaths: false },
  output: string,
): Promise<void> {
  status(`${displayName}をダウンロード`);

  const res = await fetch(url, { headers: { "Accept": kind.accept } });
  if (res.status != 200) throw new Error(`Got ${res.status}: ${url}`);
  const archiveData = new Uint8Array(await res.arrayBuffer());

  status(`${displayName}をダウンロード: 解凍中`);

  if (kind.format == "zip") {
    await extractZIP(archiveData, kind.junkPaths, output);
  } else {
    await extractTGZ(archiveData, kind.junkPaths, output);
  }

  success(`${displayName}をダウンロード: 完了`);
}

async function extractZIP(
  archiveData: Uint8Array,
  _junkPaths: true,
  output: string,
): Promise<void> {
  const zip = new ZipReader(new Uint8ArrayReader(archiveData));
  const entries = await zip.getEntries();

  await Deno.mkdir(output, { recursive: true });

  for (const entry of entries) {
    if (entry.directory) continue;
    const path = join(output, basename(entry.filename));
    const content = await entry.getData(new Uint8ArrayWriter());
    await Deno.writeFile(path, content);
  }
}

async function extractTGZ(
  archiveData: Uint8Array,
  _junkPaths: false,
  output: string,
): Promise<void> {
  const tempdir = await Deno.makeTempDir({ prefix: "download-" });
  const src = join(tempdir, "asset.tar.gz");
  await Deno.writeFile(src, archiveData);
  await tgz.uncompress(src, output);
}

function info(msg: string): void {
  console.error(`[%c*%c] %s`, "color: blue; font-weight: bold", "", msg);
}

function status(msg: string): void {
  console.error(`[%cx%c] %s`, "color: purple", "", msg);
}

function success(msg: string): void {
  console.error(`[%c+%c] %s`, "color: green; font-weight: bold", "", msg);
}

await main();
Deno.exit(0); // https://github.com/octokit/octokit.js/issues/2079
