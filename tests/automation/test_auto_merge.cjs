const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../../.github/workflows/auto-merge.yml'), 'utf8');
const marker = '          script: |';
assert.ok(source.includes(marker), 'test the actual inline workflow script');
const script = source.split(marker)[1].split(/\r?\n/).slice(1).map(line => line.startsWith('            ') ? line.slice(12) : line).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const run = new AsyncFunction('github', 'context', 'core', script);

function fixture(options = {}) {
  const calls = [];
  const pr = {
    number: 42, node_id: 'PR-test', state: 'open', draft: false,
    base: {ref: 'main'}, head: {sha: 'tested-head', repo: {full_name: 'owner/project'}},
    user: {login: 'member'}, labels: [], auto_merge: null, ...options.pr
  };
  const github = {
    rest: {
      pulls: {
        get: async () => ({data: pr}),
        merge: async payload => {
          calls.push({kind: 'merge', ...payload});
          if (options.mergeError) throw options.mergeError;
          return {data: {merged: true}};
        }
      },
      repos: {
        getCollaboratorPermissionLevel: async () => {
          calls.push({kind: 'permission'});
          if (options.permissionError) throw options.permissionError;
          return {data: {permission: options.permission || 'write'}};
        }
      }
    },
    graphql: async query => {
      const kind = query.includes('disablePullRequestAutoMerge') ? 'disable' : 'enable';
      calls.push({kind});
      if (kind === 'enable' && options.enableError) throw options.enableError;
      return {};
    }
  };
  return {calls, execute: () => run(github, {repo: {owner: 'owner', repo: 'project'}, payload: {pull_request: {number: 42}}}, {info() {}})};
}

test('a repository member is queued using native auto-merge', async () => {
  const f = fixture(); await f.execute();
  assert.deepEqual(f.calls.map(call => call.kind), ['permission', 'enable']);
});

test('a member can work from their own fork', async () => {
  const f = fixture({pr: {head: {sha: 'fork-head', repo: {full_name: 'member/fork'}}}});
  await f.execute(); assert.ok(f.calls.some(call => call.kind === 'enable'));
});

test('public contributors without write access are not auto-merged', async () => {
  const f = fixture({permission: 'read'}); await f.execute();
  assert.deepEqual(f.calls.map(call => call.kind), ['permission']);
});

test('a missing collaborator is left for manual review', async () => {
  const f = fixture({permissionError: Object.assign(new Error('not found'), {status: 404})});
  await f.execute(); assert.deepEqual(f.calls.map(call => call.kind), ['permission']);
});

test('drafts are not queued', async () => {
  const f = fixture({pr: {draft: true}}); await f.execute();
  assert.deepEqual(f.calls, []);
});

test('manual-merge pauses an existing auto-merge request', async () => {
  const f = fixture({pr: {labels: [{name: 'manual-merge'}], auto_merge: {merge_method: 'rebase'}}});
  await f.execute(); assert.deepEqual(f.calls.map(call => call.kind), ['disable']);
});

test('converting an already queued PR to draft cancels its request', async () => {
  const f = fixture({pr: {draft: true, auto_merge: {merge_method: 'squash'}}});
  await f.execute(); assert.deepEqual(f.calls.map(call => call.kind), ['disable']);
});

test('a chosen native merge method is preserved', async () => {
  const f = fixture({pr: {auto_merge: {merge_method: 'rebase'}}}); await f.execute();
  assert.deepEqual(f.calls.map(call => call.kind), ['permission']);
});

test('an already-ready merge pins the exact checked head', async () => {
  const f = fixture({enableError: new Error('Pull request is in clean status')}); await f.execute();
  assert.equal(f.calls.find(call => call.kind === 'merge').sha, 'tested-head');
});

test('native protection failures are surfaced and never bypassed', async () => {
  const f = fixture({enableError: new Error('Pull request is in clean status'), mergeError: new Error('required check failed')});
  await assert.rejects(f.execute(), /required check failed/);
  const denied = fixture({enableError: new Error('permission denied')});
  await assert.rejects(denied.execute(), /permission denied/);
  assert.ok(!denied.calls.some(call => call.kind === 'merge'));
});

test('closed PRs and other target branches are ignored', async () => {
  for (const pr of [{state: 'closed'}, {base: {ref: 'release'}}]) {
    const f = fixture({pr}); await f.execute(); assert.deepEqual(f.calls, []);
  }
});

test('the privileged workflow never checks out or runs pull-request code', () => {
  assert.ok(source.includes('pull_request_target:'));
  assert.ok(source.includes('converted_to_draft'));
  assert.ok(source.includes('labeled, unlabeled'));
  assert.ok(!/^\s*-\s+uses:\s*actions\/checkout/m.test(source));
  assert.ok(!/^\s*(?:-\s*)?run:/m.test(source));
});
